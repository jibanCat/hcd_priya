"""
Cross-check the code's `absorption_path_per_sightline` formula against the
canonical dX definition.

Canonical definition (Bahcall & Peebles 1969, Wolfe+2005 §2.3, Prochaska+
2014 §2.1, etc.):

    dX/dz = (1+z)² · H_0 / H(z)

For a snapshot box of comoving length L along the sightline, the "absorption
distance" it represents depends on the convention used to relate the box to
a redshift span.

Convention A  (proper-length box within a snapshot):
    L_phys = L_com / (1+z),   photon traverses L_phys in Δt = L_phys/c,
    |Δz_box| = (1+z) H(z) Δt = H(z) L_com / c / (1+z) × (1+z) = H(z) L_com / c
  → dX = (1+z)² H_0/H × H L_com/c = (1+z)² · L_com · H_0/c

Convention B  (naive: drop the (1+z) factor from dz_box):
    |Δz_box| = H(z) L_com / [(1+z) c]
  → dX = (1+z) · L_com · H_0/c

The code uses Convention B.  The difference is a factor of (1+z).

This test:
  1. Computes both conventions numerically at several z.
  2. Compares against the analytically-integrated X(z) formula along a sightline.
  3. Reports which convention is consistent with standard observational
     dN/dX normalisation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hcd_analysis.cddf import absorption_path_per_sightline

# Cosmology (PRIYA-like)
H0 = 70.0            # km/s/Mpc (using h = 0.7 for this test)
Om = 0.3
Ol = 0.7
c_kms = 2.99792458e5

def H(z):
    return H0 * np.sqrt(Om * (1 + z)**3 + Ol)

def dX_dz(z):
    """Canonical dX/dz."""
    return (1 + z)**2 * H0 / H(z)

def dX_box_convention_A(z, L_com_Mpc):
    """Convention A: treat sightline as a chunk of photon path through a
    box whose comoving length is fixed, with proper-length L/(1+z) at
    redshift z.  Photon crossing Δt = L/[(1+z)c] gives |Δz| = H L/c."""
    dz_box = H(z) * L_com_Mpc / c_kms
    return dX_dz(z) * dz_box

def dX_box_convention_B(z, L_com_Mpc):
    """Convention B: Δz_box = H/c · L_phys = H L_com / [(1+z) c]."""
    dz_box = H(z) * L_com_Mpc / ((1 + z) * c_kms)
    return dX_dz(z) * dz_box

def dX_code(z, L_com_Mpc):
    """What the code computes."""
    return absorption_path_per_sightline(
        box_kpc_h=L_com_Mpc * 1000 * 0.7,  # convert Mpc (h=0.7) → kpc/h
        hubble=0.7, omegam=Om, omegal=Ol, z=z,
    )

# Sanity table
print(f"\n{'z':>5}  {'conv A (1+z)² L H_0/c':>22}  {'conv B (1+z) L H_0/c':>22}  "
      f"{'ratio A/B':>10}  {'code == conv B?':>18}")
L_com = 120.0 / 0.7   # 120 Mpc/h box in Mpc
for z in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
    A = dX_box_convention_A(z, L_com)
    B = dX_box_convention_B(z, L_com)
    C = dX_code(z, L_com)
    print(f"  {z:.2f}  {A:22.5f}  {B:22.5f}  {A/B if B else float('nan'):10.3f}  "
          f"{'YES' if abs(C-B)/max(abs(B),1e-9) < 1e-6 else 'NO'}  (code={C:.5f})")

# --------------------------------------------------------------
# Check against canonical integrated X(z) over a small Δz
# --------------------------------------------------------------
print(f"\nCross-check: numerically integrate dX/dz over Δz_box = H L/c")
print(f"  expect to equal convention A's result exactly.")
from scipy.integrate import quad
z = 3.0
L_com = 120.0 / 0.7
dz_box_A = H(z) * L_com / c_kms
X_integrated, _ = quad(dX_dz, z - dz_box_A/2, z + dz_box_A/2)
print(f"  z={z}: Δz_box_A={dz_box_A:.5f},  "
      f"X_integrated over that Δz = {X_integrated:.5f},  "
      f"convention A = {dX_box_convention_A(z, L_com):.5f}")

# --------------------------------------------------------------
# Compare to what fake_spectra / PRIYA convention actually is.
# --------------------------------------------------------------
print("\nWhich convention does the LITERATURE use?")
print("  * Wolfe+2005 §2: dX = (1+z)² · H_0/H(z) · dz, photon path in comoving")
print("  * Prochaska+2014 §2.1: same, integrated over QSO-observed Δz.")
print("  * fake_spectra: applied at each snapshot.  box represents proper")
print("    chord L_com/(1+z); effective Δz = H · L_phys / c = H L_com/[(1+z)c]")
print("    → Convention B.  This is what PRIYA uses internally.")
print()
print("  => The code matches the PRIYA/fake_spectra convention (B).")
print("  => If we compare PRIYA dN/dX against observational dN/dX,")
print("     BOTH use their respective (matched) conventions, so the")
print("     comparison should be apples-to-apples IF PRIYA's assumption")
print("     that 'each sightline = 1 proper chord at z_snap' reflects what")
print("     a QSO sightline samples — i.e. observations integrate across")
print("     z, sims sum over sightlines at fixed z.")
print()
print("  => Convention A and B differ by factor (1+z). At z=3 that is 4x")
print("     = 0.60 dex. If the RIGHT convention is A but code uses B, the")
print("     CDDF/dN/dX would be inflated by (1+z).")
