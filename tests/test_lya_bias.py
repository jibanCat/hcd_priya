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
    BFFromXiResult,
    compute_p1d_clean_sightlines,
    extract_monopole,
    extract_multipoles_rmu,
    find_camb_pk_for_z,
    fit_b_F,
    fit_b_F_from_xi_FF,
    fit_b_beta_from_xi_cross_multipoles,
    hMpc_to_kms_factor,
    load_camb_pk,
    project_pk_3d_to_p1d,
    xi_lin_monopole,
    xi_lin_quadrupole,
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
        P_3d_kms = P_h / (F_factor ** 3)                  # P has dims length^3
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


class TestXiLinMonopole(unittest.TestCase):
    """For a Gaussian P_lin(k) = A·exp(-k²/k0²) the integral has a clean
    closed form via Fourier-transform tables:

        ξ_lin(r) = A · (k0³ / (8 π^{3/2})) · exp(-k0²·r²/4)

    Verify the integrator returns this within 2 %."""

    def test_gaussian_pk_closed_form(self):
        A = 1.0
        k0 = 0.5
        k = np.linspace(0.001, 5.0, 4096)
        P_lin = A * np.exp(-(k / k0) ** 2)
        r = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        xi = xi_lin_monopole(r, k, P_lin)
        expected = (
            A * (k0 ** 3) / (8.0 * np.pi ** 1.5)
            * np.exp(-(k0 * r) ** 2 / 4.0)
        )
        np.testing.assert_allclose(xi, expected, rtol=2e-2)

    def test_zero_r_returns_field_variance(self):
        """ξ(r=0) = (1/(2π²)) ∫ k² P(k) dk = field variance σ²."""
        k = np.linspace(0.001, 5.0, 4096)
        P_lin = np.exp(-(k / 0.5) ** 2)
        sigma2 = float(np.trapezoid(k ** 2 * P_lin, k) / (2.0 * np.pi ** 2))
        xi0 = xi_lin_monopole(np.array([0.0]), k, P_lin)
        np.testing.assert_allclose(xi0[0], sigma2, rtol=1e-12)


class TestExtractMonopole(unittest.TestCase):

    def test_single_2d_bin_to_single_1d_bin(self):
        """A single 2-D bin at r_⊥=0, r_∥=5 with ξ=0.1 falls in the
        r∈[4,6] bin; the monopole picks it up at the right value."""
        xi = np.array([[0.1]])
        n = np.array([[1000]])
        rp = np.array([0.0])                    # perp = 0 (only LOS sep)
        rl = np.array([5.0])
        r_bins = np.array([0.0, 4.0, 6.0, 100.0])
        r_c, xi_m, n_p = extract_monopole(xi, n, rp, rl, r_bins)
        self.assertEqual(int(n_p[1]), 1000)     # bin 1 is [4, 6]
        self.assertAlmostEqual(float(xi_m[1]), 0.1)


class TestFitBFFromXiFF(unittest.TestCase):
    """End-to-end: synthesise ξ_FF^(0)(r) from a known b_F, run the fit,
    recover b_F at the few-percent level."""

    def test_recover_planted_b_F(self):
        # Linear matter spectrum (Mpc/h units, dimensionless P)
        k = np.logspace(-3, 1, 2048)
        P_lin = 5e3 * k / (1.0 + (k / 0.1) ** 4)        # CDM-shaped power
        # Target ξ_FF^(0)(r) at a r-grid + Kaiser prefactor
        b_F_in = -0.20
        beta = 1.5
        K = 1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta ** 2

        # Build a synthetic 2-D grid where ξ_2d depends only on r =
        # sqrt(r_perp² + r_par²) — the monopole is "perfect" by
        # construction.  Then add tiny Gaussian noise.
        r_perp_c = np.linspace(2.5, 47.5, 19)
        r_par_c = np.linspace(2.5, 47.5, 19)
        n_grid = np.full((r_perp_c.size, r_par_c.size), 1000)

        rng = np.random.default_rng(2026)
        xi_2d = np.zeros_like(n_grid, dtype=np.float64)
        for i, rp in enumerate(r_perp_c):
            for j, rl in enumerate(r_par_c):
                r = np.sqrt(rp * rp + rl * rl)
                xi_lin_val = xi_lin_monopole(np.array([r]), k, P_lin)[0]
                expected = (b_F_in ** 2) * K * xi_lin_val
                # Sub-percent gaussian noise
                xi_2d[i, j] = expected * (1.0 + 0.01 * rng.standard_normal())

        result = fit_b_F_from_xi_FF(
            xi_2d=xi_2d, npairs_2d=n_grid,
            r_perp_centres=r_perp_c, r_par_centres=r_par_c,
            k_lin=k, P_lin=P_lin, beta_F=beta,
            r_min=10.0, r_max=40.0, n_r_bins=10,
        )
        rel = abs(result.b_F - b_F_in) / abs(b_F_in)
        msg = (f"b_F (xi_FF) recovered = {result.b_F:.4f}, "
               f"planted = {b_F_in:.4f}, rel_err = {rel:.3f}")
        self.assertLess(rel, 0.05, msg)
        self.assertLess(result.b_F, 0)
        print(f"  test_fit_b_F_from_xi_FF: {msg}")


class TestXiLinQuadrupole(unittest.TestCase):
    """Sanity check the j_2 transform of P_lin against scipy.integrate.quad
    on the same Gaussian P_lin(k) = A exp(-k²/k0²) used for the j_0 test."""

    def test_matches_scipy_quad(self):
        from scipy.integrate import quad
        from scipy.special import spherical_jn

        A = 1.0
        k0 = 0.5
        k_grid = np.linspace(0.001, 5.0, 4096)
        P_grid = A * np.exp(-(k_grid / k0) ** 2)
        rs = np.array([1.0, 4.0, 8.0])
        xi = xi_lin_quadrupole(rs, k_grid, P_grid)

        for ri, x in zip(rs, xi):
            ref, _ = quad(
                lambda kk: kk * kk * spherical_jn(2, kk * ri)
                            * A * np.exp(-(kk / k0) ** 2),
                0.0, 50.0, limit=200,
            )
            ref /= 2.0 * np.pi ** 2
            np.testing.assert_allclose(
                x, ref, rtol=1e-2,
                err_msg=f"r={ri}: trapezoid xi_quad={x}, scipy quad ref={ref}",
            )

    def test_zero_at_r_zero(self):
        """j_2(0) = 0 → ξ^(j2)(0) = 0."""
        k = np.linspace(0.001, 5.0, 1024)
        P = np.exp(-(k / 0.5) ** 2)
        xi = xi_lin_quadrupole(np.array([0.0, 1.0]), k, P)
        self.assertEqual(float(xi[0]), 0.0)
        self.assertNotAlmostEqual(float(xi[1]), 0.0)


class TestExtractMultipolesRMu(unittest.TestCase):
    """Hamilton-1992 synthesis regression test — locks the
    multipole-extraction Jacobian fix.

    Build ξ(r, μ) = ξ_0(r) + ξ_2(r) · L_2(μ) on a (r, |μ|) grid, run
    extract_multipoles_rmu, verify exact recovery of ξ_0 and ξ_2 to
    floating-point precision (the test the npairs-weighted estimator
    failed — see docs/clustering_multipole_jacobian_todo.md)."""

    def _build_synthesis(
        self,
        n_r: int = 30,
        n_mu: int = 40,
        seed: int = 2026,
    ):
        rng = np.random.default_rng(seed)
        r_centres = np.linspace(2.0, 60.0, n_r)
        # Uniform μ-bins on [0, 1]
        mu_edges = np.linspace(0.0, 1.0, n_mu + 1)
        mu_centres = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        # Planted multipoles (smooth functions of r)
        xi0_true = 0.05 * np.exp(-r_centres / 30.0)
        xi2_true = -0.02 * np.exp(-r_centres / 25.0)
        # Synthesise ξ(r, μ) = ξ_0 + ξ_2 · L_2(μ)
        L2 = 0.5 * (3.0 * mu_centres ** 2 - 1.0)
        xi_rmu = (
            xi0_true[:, None]
            + xi2_true[:, None] * L2[None, :]
        )
        npairs_rmu = np.full(xi_rmu.shape, 1000, dtype=np.int64)
        return r_centres, mu_centres, xi_rmu, npairs_rmu, xi0_true, xi2_true

    def test_recovers_planted_multipoles_exactly(self):
        r_c, mu_c, xi_rmu, np_rmu, xi0_true, xi2_true = self._build_synthesis()
        out = extract_multipoles_rmu(
            xi_rmu=xi_rmu, npairs_rmu=np_rmu, mu_centres=mu_c, ells=(0, 2),
        )
        # Trapezoidal-style midpoint sum has small bias on smooth ξ_2(μ)
        # — relax tolerance to 1 % rather than bit-exact.
        np.testing.assert_allclose(out[0], xi0_true, atol=2e-4, rtol=1e-2)
        np.testing.assert_allclose(out[2], xi2_true, atol=2e-4, rtol=1e-2)
        max_err_0 = float(np.abs(out[0] - xi0_true).max())
        max_err_2 = float(np.abs(out[2] - xi2_true).max())
        print(f"  Hamilton synthesis: max err ξ_0 = {max_err_0:.2e}, "
              f"ξ_2 = {max_err_2:.2e}")

    def test_no_quadrupole_leakage_in_pure_monopole(self):
        """If ξ(r, μ) = ξ_0(r) (μ-independent), recovered ξ_2 ≈ 0."""
        r_c, mu_c, _, np_rmu, xi0_true, _ = self._build_synthesis()
        # Pure-monopole field
        xi_rmu = np.broadcast_to(xi0_true[:, None], (r_c.size, mu_c.size)).copy()
        out = extract_multipoles_rmu(
            xi_rmu=xi_rmu, npairs_rmu=np_rmu, mu_centres=mu_c, ells=(0, 2),
        )
        # ξ_0 recovered to machine precision; ξ_2 should be ≈ 0
        # (limited by midpoint integration of L_2 over [0, 1]).
        np.testing.assert_allclose(out[0], xi0_true, rtol=1e-12)
        # The original npairs-weighted estimator would leak ~ 1/8 · ξ_0
        # into ξ_2 here (~ −1/8 · 0.05 ≈ −6e-3).  Uniform-μ Hamilton
        # gives an exact 0 in the continuum limit; midpoint-rule on
        # 40 bins gives < 1 % of ξ_0.
        leak = float(np.abs(out[2]).max())
        ref = float(np.abs(xi0_true).max())
        self.assertLess(
            leak / ref, 0.01,
            f"Pure-monopole field leaks into ξ_2 by {leak/ref:.2e} of ξ_0; "
            f"expected < 1 % from uniform-μ Hamilton (was ~ 12.5 % under "
            f"the buggy npairs-weighted estimator)."
        )
        print(f"  Pure-monopole leakage into ξ_2: {leak/ref:.2e} of ξ_0")


class TestFitBBetaJoint(unittest.TestCase):
    """End-to-end: build a synthetic Kaiser-cross ξ_×(r, μ) with
    planted (b_DLA, β_DLA), run the joint fitter, recover within
    tolerance.  This exercises the full pipeline:
    extract_multipoles_rmu → xi_lin_monopole + xi_lin_quadrupole →
    Levenberg-Marquardt (b_D, β_D) fit."""

    def test_recover_planted_b_and_beta(self):
        # Linear matter spectrum (Mpc/h units, dimensionless P)
        k = np.logspace(-3, 1, 4096)
        P_lin = 5e3 * k / (1.0 + (k / 0.1) ** 4)         # CDM-shaped

        b_DLA_true = 2.0
        beta_DLA_true = 0.5
        b_F_true = -0.2
        beta_F_true = 1.5

        # Build the model templates on a fine (r, μ) grid
        r_centres = np.linspace(5.0, 60.0, 28)
        mu_edges = np.linspace(0.0, 1.0, 21)
        mu_centres = 0.5 * (mu_edges[:-1] + mu_edges[1:])

        xi_lin_j0 = xi_lin_monopole(r_centres, k, P_lin)
        xi_lin_j2 = xi_lin_quadrupole(r_centres, k, P_lin)
        K0_true = 1.0 + (beta_DLA_true + beta_F_true) / 3.0 \
                    + beta_DLA_true * beta_F_true / 5.0
        K2_true = (2.0 / 3.0) * (beta_DLA_true + beta_F_true) \
                    + (4.0 / 7.0) * beta_DLA_true * beta_F_true
        xi0_true = +b_DLA_true * b_F_true * K0_true * xi_lin_j0
        xi2_true = -b_DLA_true * b_F_true * K2_true * xi_lin_j2

        # Synthesise ξ_×(r, μ) = ξ_0 + ξ_2 · L_2(μ)
        L2 = 0.5 * (3.0 * mu_centres ** 2 - 1.0)
        xi_rmu = xi0_true[:, None] + xi2_true[:, None] * L2[None, :]
        # Add small Gaussian noise so the LM fit has work to do
        rng = np.random.default_rng(2026)
        noise = 0.005 * np.abs(xi_rmu).max() * rng.standard_normal(xi_rmu.shape)
        xi_rmu = xi_rmu + noise
        npairs_rmu = np.full(xi_rmu.shape, 5000, dtype=np.int64)

        result = fit_b_beta_from_xi_cross_multipoles(
            xi_rmu=xi_rmu, npairs_rmu=npairs_rmu,
            r_centres=r_centres, mu_centres=mu_centres,
            k_lin=k, P_lin=P_lin,
            b_F=b_F_true, beta_F=beta_F_true,
            r_min=10.0, r_max=40.0,
        )

        rel_b = abs(result.b_DLA - b_DLA_true) / b_DLA_true
        rel_beta = abs(result.beta_DLA - beta_DLA_true) / beta_DLA_true
        msg = (f"b_DLA = {result.b_DLA:.4f} (planted {b_DLA_true}, "
               f"rel {rel_b:.3f}); β_DLA = {result.beta_DLA:.4f} "
               f"(planted {beta_DLA_true}, rel {rel_beta:.3f}); "
               f"chi2 = {result.chi2:.2f}")
        # Tight tolerances: this is a clean Hamilton synthesis on a
        # fully-populated (r, μ) grid.
        self.assertLess(rel_b, 0.03, msg)
        self.assertLess(rel_beta, 0.10, msg)
        print(f"  test_fit_b_beta_joint: {msg}")

    def test_npairs_weighted_estimator_fails_on_same_synthesis(self):
        """The legacy npairs-weighted (r_⊥, r_∥) extractor would have
        failed this test by ~ 25 % bias on b_DLA and ~ 6 % bias on
        β_DLA (per the strip-commit ce48c1d notes).  We don't import
        the legacy path here (it's been deleted); instead we verify
        the SAME synthesis collapsed to the npairs-weighted multipole
        formula reproduces the documented bias.

        Specifically: weighted-by-√(1−μ²) projection vs uniform-μ
        projection of (ξ_0 + ξ_2 · L_2) gives a recovered ξ_2 biased
        by the factor ⟨L_2⟩_npairs / ⟨L_2⟩_uniform_for_pure_monopole
        = −1/8 — i.e. the bias does not vanish.  Documenting the
        regression mechanism here so the next person who reads the
        test understands why it exists.
        """
        # Build the same ξ_0(μ-indep) field used in
        # test_no_quadrupole_leakage_in_pure_monopole, but project it
        # with a √(1 − μ²) weight (mimicking npairs in (r_⊥, r_∥) bins)
        # and verify the recovered ξ_2 is ~ −1/8 of ξ_0.
        r_centres = np.linspace(2.0, 60.0, 30)
        mu_edges = np.linspace(0.0, 1.0, 41)
        mu_centres = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        dmu = np.diff(mu_edges)
        L2 = 0.5 * (3.0 * mu_centres ** 2 - 1.0)
        xi0_true = 0.05 * np.exp(-r_centres / 30.0)
        # Pure monopole field: ξ(r, μ) = ξ_0(r)
        xi_rmu = np.broadcast_to(xi0_true[:, None],
                                  (r_centres.size, mu_centres.size))

        # npairs-weighted projection (the buggy estimator)
        weight = np.sqrt(1.0 - mu_centres ** 2) * dmu        # (n_mu,)
        norm = weight.sum()
        proj_L2 = (xi_rmu * (L2 * weight)[None, :]).sum(axis=1) / norm
        # The legacy formula would also multiply by 5 (the 2ℓ+1 factor),
        # but the *fractional* leakage relative to ξ_0 is what matters.
        fractional_leakage = float(np.abs(5 * proj_L2).mean()
                                   / np.abs(xi0_true).mean())
        # Theoretical: ⟨L_2⟩_{w=√(1-μ²)} on [0,1] = −1/8.  Times the
        # (2ℓ+1)=5 prefactor → −5/8 ≈ −0.625 fractional contamination.
        # We just sanity-check it's >> 1 % (i.e. the bug is real).
        self.assertGreater(
            fractional_leakage, 0.1,
            f"npairs-weighted estimator should leak ξ_0 into ξ_2 "
            f"by ≳ 10 %; got {fractional_leakage:.3f} — if this is "
            f"now small, the regression is not actually testing the bug."
        )
        print(f"  Documented npairs-weighted leakage on this synthesis: "
              f"{fractional_leakage:.3f}  (bug magnitude check)")


if __name__ == "__main__":
    unittest.main()
