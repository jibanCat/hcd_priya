# Benchmarking

Run `hcd benchmark` to populate this file with actual timing results.

## Extrapolation method

1. Run catalog build and P1D on 10k skewers (≈1.5% of total).
2. Scale timing by `n_total_skewers / 10000`.
3. Multiply by number of (sim, snap) pairs.
4. Divide by n_workers for parallel estimate.

## Expected scale

With 60 sims × ~18 snapshots per sim = ~1080 (sim, snap) pairs,
and ~4–8 SLURM nodes (each 36–48 cores):

- Fast mode (no Voigt fitting): estimated 2–6 hours wall time.
- Full Voigt fitting: estimated 12–48 hours depending on hardware.
