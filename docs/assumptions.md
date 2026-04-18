# Explicit Assumptions

## Data

1. **tau is raw simulation optical depth** — not rescaled to any mean flux.
   The UV background is the simulation's native UVB; no post-processing
   mean-flux rescaling is applied before P1D or CDDF computation.

2. **colden is empty** — the `colden` HDF5 group key exists but contains no
   datasets. All column density estimates are derived from `tau/H/1/1215`.

3. **Only Lyman-alpha tau** — only `tau/H/1/1215` (HI 1216 Å) is used.
   No higher Lyman series, no metal lines.

4. **Primary file is grid_480** — 691200 sightlines on a 480×480×3 grid.
   When absent, falls back to `lya_forest_spectra.hdf5` (32000 sightlines).

5. **Snapshots.txt is authoritative** for (snap, a) mapping. If a SPECTRA
   folder exists without a matching Snapshots.txt entry, the redshift is
   read from the HDF5 Header.

## Absorber identification

6. **System = connected tau region** — an absorption system is defined as a
   contiguous set of pixels where `tau > tau_threshold` (default: 1.0).
   This corresponds approximately to log10(NHI) > 16 for a typical b=30 km/s.

7. **Merging criterion** — two adjacent blobs are merged into one system if
   the gap between them is < `merge_dv_kms` (default: 100 km/s) in velocity
   space. This prevents artificial splitting of DLAs with narrow sub-structure.

8. **NHI via Voigt fitting** — NHI is estimated by fitting a single Voigt
   component (2 parameters: NHI, b) to each system's tau profile.
   In fast/benchmark mode, a closed-form approximation is used instead.

9. **Classification thresholds** (standard literature values):
   - LLS    : 10^17.2 ≤ NHI < 10^19.0 cm^-2
   - subDLA : 10^19.0 ≤ NHI < 10^20.3 cm^-2
   - DLA    : NHI ≥ 10^20.3 cm^-2

## P1D

10. **Mean flux normalisation** — delta_F = F/<F> - 1 where <F> is computed
    globally over all skewers in the file (not per-skewer).

11. **No mean flux rescaling** — we do not rescale tau to match observed mean
    flux. If rescaling is needed, apply it externally before running this pipeline.

12. **k units** — k is in s/km (inverse velocity). Conversion to h/Mpc:
    k [h/Mpc] = k [s/km] × H(z)/h × 1000.

13. **Box boundary** — sightlines are periodic (from MP-Gadget). No windowing
    is applied to the FFT.

## CDDF

14. **Absorption path** — uses the narrow-box approximation:
    dX = (1+z) × L_comoving_Mpc × H_0/c.

15. **NHI bins** — log10(NHI) from 17 to 23 in 30 bins by default.

## General

16. **Cosmology is per-sim** — omega_m, omega_l, h are read from the HDF5 Header
    for each file. There is no global assumed cosmology.

17. **Box size is fixed** — all sims use a 120 Mpc/h comoving box.

18. **No peculiar velocity correction** — the tau array from fake_spectra
    already includes peculiar velocity contributions (it is computed in
    redshift space). No correction is applied.
