# CDDF Model and Perturbation

## Definition

The column density distribution function (CDDF) is:

    f(N_HI, X) = d^2 n / (dN_HI dX)

where:
  - N_HI: neutral hydrogen column density (cm^-2)
  - X: dimensionless absorption path length
  - n: number of absorbers per sightline

## Absorption path length

For a flat ΛCDM cosmology:

    dX/dz = (H_0 / H(z)) × (1+z)^2

For a simulation box at redshift z with comoving size L [Mpc]:

    dX = (H_0 / H(z)) × (1+z)^2 × dz_box

where dz_box = H(z)/c × L_phys = H(z)/c × L/(1+z), giving:

    dX = (1+z) × L × H_0/c

Per sightline: dX = (1+z) × (box_Mpc) × (H_0/c)
Total path: dX_total = N_sightlines × dX

## Measurement

From the absorber catalog, we bin absorbers by log10(NHI) and compute:

    f(N_i) = count(N in bin i) / (dN_i × dX_total)

where dN_i = 10^(log_N_max) - 10^(log_N_min) in cm^-2.

## Perturbation model

We define a continuous multiplicative perturbation:

    f'(N) = A × f(N) × (N / N_pivot)^alpha

Parameters:
  A       : amplitude multiplier       (default 1.0, i.e. no change)
  alpha   : power-law tilt             (default 0.0)
  N_pivot : pivot column density (cm^-2, default 10^20)

Special cases:
  - A=1, alpha=0: no perturbation (identity)
  - A>1, alpha=0: uniform increase of all absorbers
  - alpha>0: tilt toward high-NHI systems (more DLAs relative to LLS)
  - alpha<0: tilt toward low-NHI systems

## Propagation to P1D

To compute the P1D effect of the CDDF perturbation, we use Poisson resampling:

1. For each absorber i with column density N_i, compute weight:
       w_i = A × (N_i / N_pivot)^alpha

2. Draw n_copies_i ~ Poisson(w_i) for each absorber.

3. Build a perturbed absorber set from this resampling.

4. Compute masked P1D with the perturbed absorber set as the mask.

5. Repeat for N_realizations to estimate Monte Carlo variance.

The ratio P1D_perturbed / P1D_baseline quantifies the P1D correction
due to the CDDF perturbation.

## Interpretation

This model allows the HCD correction to the P1D to be continuously tunable:
- The amplitude A shifts the overall absorber number density.
- The tilt alpha changes the relative contributions of LLS, subDLA, and DLA
  to the P1D suppression.
- The pivot N_pivot sets where the tilt has no effect (f'(N_pivot) = A × f(N_pivot)).

For emulator purposes, the combination (A, alpha) at fixed N_pivot provides
a 2-parameter family of CDDF corrections that can be marginalised over.
