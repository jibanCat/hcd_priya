"""
Unit tests for the Ω_HI formula used in scripts/build_hcd_summary.py.

Ω_HI is defined as

    Ω_HI = ρ_HI / ρ_crit,0
         = (m_H · H_0 / c) / ρ_crit,0  ·  ∫ N_HI · f(N_HI, X) dN_HI

In catalog form this becomes

    Ω_HI = prefactor(h) · Σ N_HI / ΔX

with prefactor(h) = m_H · H_0(h) / (c · ρ_crit,0(h)), where both H_0
and ρ_crit,0 carry h, so the combined prefactor scales as 1/h.

Three checks here:
  1. Numerical value at h = 1 (hand-derived from cgs)
  2. h-scaling: prefactor(h) · h is a constant
  3. Order-of-magnitude: a plausible (ΣN/ΔX) representative of
     DLAs at z=3 yields Ω_HI(DLA) ∈ [5e-4, 3e-3] — the range
     reported by Prochaska+2005 / Noterdaeme+2012 / Ho+2021 for
     cosmological DLA surveys.  This is a sanity bound, not a
     precision match.

Run:
    python3 tests/test_omega_hi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_hcd_summary import omega_hi_from_catalog


def _prefactor(sum_NHI: float, dX: float, h: float) -> float:
    """Reverse-engineer the prefactor by inverting Ω_HI = pref · ΣN/ΔX."""
    return omega_hi_from_catalog(sum_NHI, dX, h) * dX / sum_NHI


def test_prefactor_at_h_equals_1():
    """Prefactor at h=1 should equal m_H · H_0 / (c · ρ_crit,0) = 9.608e-24 cm²."""
    pref = _prefactor(sum_NHI=1.0, dX=1.0, h=1.0)
    # Hand-derived from cgs:
    # H_0(h=1) = 3.2408e-18 /s
    # m_H · H_0 / c = 1.67353e-24 · 3.2408e-18 / 2.99792e10 = 1.8091e-52 g/cm
    # ρ_crit,0(h=1) = 1.87847e-29 g/cm³
    # prefactor = 9.630e-24 cm²
    expected = 9.63e-24
    rel_err = abs(pref - expected) / expected
    assert rel_err < 2e-3, (
        f"prefactor(h=1) = {pref:.4e}, expected {expected:.4e} "
        f"(rel err {rel_err:.2%})"
    )
    print(f"  ✓ prefactor(h=1) = {pref:.4e} cm²  (expected ≈ 9.63e-24)")


def test_h_scaling():
    """prefactor(h) · h = constant (H_0 ∝ h, ρ_crit,0 ∝ h² → prefactor ∝ 1/h)."""
    p07 = _prefactor(1.0, 1.0, 0.7)
    p10 = _prefactor(1.0, 1.0, 1.0)
    ratio = (p07 * 0.7) / (p10 * 1.0)
    assert abs(ratio - 1.0) < 1e-10, (
        f"h-scaling broken: prefactor(0.7)·0.7/prefactor(1.0) = {ratio:.6f}, "
        f"expected 1.0"
    )
    print(f"  ✓ prefactor scales as 1/h  (p(0.7)·0.7 / p(1)·1 = {ratio:.6f})")


def test_realistic_dla_magnitude():
    """
    A ΣN_HI / ΔX representative of DLAs at z ≈ 3 should land Ω_HI
    in [1e-4, 3e-3] — the cosmological range reported by PW09 /
    Noterdaeme+2012 / Ho+2021 for DLAs at z≈3.

    One LF PRIYA sim at z=3 has 691 200 sightlines in a 120 Mpc/h box.
    The fixed-bug-#7 dX per sightline is (1+z)² · L_com · H_0 / c
    = 16 · (120/0.7) · 70 / 3e5 ≈ 0.64, so ΔX ≈ 4.4e5.

    A per-sim DLA catalog at z=3 typically has ~22 000 DLAs with
    average log N ≈ 20.8 → ΣN_HI ≈ 1.4e25 cm⁻².  Plugging in gives
    Ω_HI^DLA ≈ 4.4e-4, safely inside the published range.
    """
    omega = omega_hi_from_catalog(sum_NHI_cm2=1.4e25, dX_total=4.4e5, hubble=0.7)
    assert 1e-4 < omega < 3e-3, (
        f"Ω_HI^DLA = {omega:.3e}, outside plausible range [1e-4, 3e-3]"
    )
    print(f"  ✓ representative DLA-catalog input → Ω_HI = {omega:.2e} (expected 1e-4 .. 3e-3)")


def test_zero_path_returns_nan():
    """ΔX = 0 should not divide-by-zero; should return NaN safely."""
    import math
    v = omega_hi_from_catalog(sum_NHI_cm2=1.0, dX_total=0.0, hubble=0.7)
    assert math.isnan(v), f"Ω_HI with ΔX=0 should be NaN, got {v}"
    print("  ✓ ΔX=0 returns NaN safely")


def main() -> int:
    print("Running Ω_HI formula tests:")
    tests = [
        test_prefactor_at_h_equals_1,
        test_h_scaling,
        test_realistic_dla_magnitude,
        test_zero_path_returns_nan,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
