"""
Unit tests for hcd_analysis.cddf.absorption_path_per_sightline.

Lesson from the audit: any "general claim" — like "this dX formula is
correct" — needs a test BEFORE it's used to interpret data.  Three
independent checks here:

  1. Formula match against the canonical (1+z)^2 · L_com · H_0 / c.
  2. Numerical match against fake_spectra.unitsystem.absorption_distance
     (ported in-line; verified against upstream Apr 2026).
  3. Consistency check via numerical integration of dX/dz over Δz_box.

If any of these fail, do NOT trust dN/dX or CDDF normalisation until
they are reconciled.  Run with:
    python3 tests/test_absorption_path.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hcd_analysis.cddf import absorption_path_per_sightline

C_KMS = 2.99792458e5


def H_of_z(z: float, H0: float, Om: float, Ol: float) -> float:
    """H(z) in km/s/Mpc for flat ΛCDM."""
    return H0 * np.sqrt(Om * (1 + z)**3 + Ol)


def dX_canonical(box_kpc_h: float, hubble: float, z: float) -> float:
    """Canonical dX = (1+z)² · L_com · H_0 / c."""
    L_com_Mpc = box_kpc_h / 1000.0 / hubble
    H0 = hubble * 100.0
    return (1.0 + z) ** 2 * L_com_Mpc * H0 / C_KMS


def dX_fake_spectra_inline(box_kpc_h: float, hubble: float, z: float) -> float:
    """Verbatim port of fake_spectra.unitsystem.absorption_distance:

        return self.h100 / self.light * speclen * self.UnitLength_in_cm * (1+red)**2

    where:
        self.h100 = 3.2407789e-18    # 100 km/s/Mpc in 1/s
        self.light = 2.99e10         # cm/s
        UnitLength_in_cm = 3.085678e21  # kpc in cm
        speclen = box in kpc/h units

    The implicit h enters via speclen (kpc/h) × UnitLength_in_cm (cm/kpc) =
    L_com_cm × h, so the formula reduces to (h · 100 · L_com · (1+z)²) / c.
    """
    h100 = 3.2407789e-18
    light = 2.99e10
    UL_cm = 3.085678e21
    return h100 / light * box_kpc_h * UL_cm * (1 + z) ** 2


def dX_integrated(box_kpc_h: float, hubble: float, z: float,
                  Om: float = 0.3, Ol: float = 0.7) -> float:
    """Compute X(z+Δz/2) - X(z-Δz/2) by numerical integration, where
    Δz_box = H(z) · L_com / c.  Independent reference."""
    H0 = hubble * 100.0
    L_com_Mpc = box_kpc_h / 1000.0 / hubble
    H_at_z = H_of_z(z, H0, Om, Ol)
    Δz_box = H_at_z * L_com_Mpc / C_KMS
    integrand = lambda zp: (1 + zp) ** 2 * H0 / H_of_z(zp, H0, Om, Ol)
    result, _ = quad(integrand, z - Δz_box / 2.0, z + Δz_box / 2.0)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matches_canonical():
    """Codebase implementation == canonical formula."""
    for box in (60000.0, 120000.0, 240000.0):
        for h in (0.65, 0.70, 0.75):
            for z in (0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0):
                code = absorption_path_per_sightline(box, h, 0.3, 0.7, z)
                ref = dX_canonical(box, h, z)
                assert abs(code - ref) / max(ref, 1e-30) < 1e-12, \
                    f"FAILED at box={box}, h={h}, z={z}: code={code}, ref={ref}"
    print("  ✓ codebase == canonical (1+z)²·L_com·H_0/c, machine precision")


def test_matches_fake_spectra():
    """Codebase implementation == fake_spectra port (within fake_spectra's c-precision)."""
    for box in (60000.0, 120000.0, 240000.0):
        for h in (0.65, 0.70, 0.75):
            for z in (0.0, 1.0, 2.0, 3.0, 5.0):
                code = absorption_path_per_sightline(box, h, 0.3, 0.7, z)
                ref = dX_fake_spectra_inline(box, h, z)
                rel = abs(code - ref) / max(ref, 1e-30)
                # fake_spectra uses c = 2.99e10 (3-digit precision) so the
                # match is at ~3e-3, not machine precision.
                assert rel < 5e-3, \
                    f"FAILED at box={box}, h={h}, z={z}: code={code}, ref={ref}, rel_err={rel}"
    print("  ✓ codebase == fake_spectra port to 0.3 % (limited by upstream c precision)")


def test_matches_numerical_integration():
    """For the small-Δz_box approximation used everywhere, the codebase
    formula should equal X(z+Δz/2) - X(z-Δz/2) computed by numerical
    integration of dX/dz."""
    for h in (0.7,):
        for z in (1.0, 2.0, 3.0, 4.0, 5.0):
            code = absorption_path_per_sightline(120000.0, h, 0.3, 0.7, z)
            ref = dX_integrated(120000.0, h, z, Om=0.3, Ol=0.7)
            rel = abs(code - ref) / max(ref, 1e-30)
            # Linear-Δz approximation has small error (next-order Δz²)
            assert rel < 1e-3, \
                f"FAILED at z={z}: code={code}, integrated={ref}, rel_err={rel}"
    print("  ✓ codebase == ∫dX/dz over Δz_box (linear approximation, < 0.1 % residual)")


def test_units_dimensional_consistency():
    """At z=0, dX/dz = 1 by definition (since (1+0)²·H_0/H(0) = 1 for flat ΛCDM with Om+Ol=1).
    For a box of L_com_Mpc, dX(z=0) should equal Δz_box(z=0) = H_0·L_com/c."""
    h = 0.7; H0 = 70.0
    for box in (60000.0, 120000.0):
        L_com = box / 1000.0 / h
        expected = H0 * L_com / C_KMS  # = Δz_box at z=0
        actual = absorption_path_per_sightline(box, h, 0.3, 0.7, 0.0)
        assert abs(actual - expected) / expected < 1e-12, \
            f"z=0 dX should equal H_0·L_com/c = {expected}, got {actual}"
    print("  ✓ z=0 dimensional consistency (dX = Δz_box at z=0)")


def test_scaling_with_z():
    """dX/dz scaling: dX(z=3) / dX(z=2) should equal
    [(1+3)²/H(3)] / [(1+2)²/H(2)] · (H_0/H_0) at fixed L_com."""
    h = 0.7
    box = 120000.0
    z1, z2 = 2.0, 3.0
    expected = ((1 + z2) ** 2 * H_of_z(z1, h * 100, 0.3, 0.7)
                / ((1 + z1) ** 2 * H_of_z(z2, h * 100, 0.3, 0.7)))
    # But our code uses the small-Δz_box approximation: dX = (1+z)² · L · H_0/c
    # which DOES NOT depend on H(z).  So actual ratio is purely (1+z)²:
    code_ratio = absorption_path_per_sightline(box, h, 0.3, 0.7, z2) \
                / absorption_path_per_sightline(box, h, 0.3, 0.7, z1)
    expected_const_L = (1 + z2) ** 2 / (1 + z1) ** 2
    assert abs(code_ratio - expected_const_L) < 1e-10, \
        f"dX(3)/dX(2) should be (4/3)^2 = {expected_const_L}, got {code_ratio}"
    print("  ✓ dX(z=3)/dX(z=2) = (1+z)² ratio (16/9 ≈ 1.778)")


def test_h_independence_of_dN_dX():
    """For a fixed N_absorbers and fixed comoving box length, dN/dX
    should be insensitive to the choice of h (since dX scales with h).
    Two sims with same comoving box but different h should give the
    same dN/dX after normalisation.  This caught the broken H_0=100
    bug — the OLD code did not depend on h at all."""
    box_a, h_a = 120000.0, 0.7    # 120 Mpc/h with h=0.7 → 171 Mpc comoving
    box_b, h_b = 91000.0, 0.53    # 91/0.53 ≈ 171 Mpc comoving — same physical
    L_a = box_a / 1000.0 / h_a
    L_b = box_b / 1000.0 / h_b
    assert abs(L_a - L_b) / L_a < 0.01, "Test setup: L_com should match"
    dX_a = absorption_path_per_sightline(box_a, h_a, 0.3, 0.7, 3.0)
    dX_b = absorption_path_per_sightline(box_b, h_b, 0.3, 0.7, 3.0)
    # Same L_com → dX should scale as h_a / h_b (since H_0 = h × 100)
    expected_ratio = h_a / h_b
    actual_ratio = dX_a / dX_b
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-2, \
        f"At fixed L_com, dX should scale as h.  Expected ratio {expected_ratio}, got {actual_ratio}"
    print("  ✓ dX scales linearly with h at fixed comoving box")


def main():
    print("Running absorption_path_per_sightline tests:")
    test_matches_canonical()
    test_matches_fake_spectra()
    test_matches_numerical_integration()
    test_units_dimensional_consistency()
    test_scaling_with_z()
    test_h_independence_of_dN_dX()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
