# P1D Definition

## Flux field

Given the optical depth tau(v) along a sightline (pixel index j, velocity v_j = j × dv):

    F(v) = exp(-tau(v))

The fractional flux fluctuation is:

    delta_F(v) = F(v) / <F> - 1

where <F> is the mean transmission averaged over all N_skewers sightlines:

    <F> = (1 / N_skewers / N_pix) × sum_{all pixels} exp(-tau_ij)

## Fourier transform convention

We use the discrete Fourier transform (DFT) along the velocity axis:

    tilde_F(k_n) = dv × sum_{j=0}^{N-1} delta_F(v_j) × exp(-2 pi i k_n v_j)

where:
    k_n = n / (N × dv)    for n = 0, 1, ..., N/2

Units: tilde_F has units of km/s.

## Power spectrum

    P1D(k_n) = (1 / L) × |tilde_F(k_n)|^2

where L = N × dv is the total velocity extent of the box (km/s).

Substituting:

    P1D(k_n) = dv / N × |DFT(delta_F)[n]|^2

Units: [P1D] = km/s.

In code: `P1D[n] = dv / N * |rfft(delta_F * dv)|^2 / dv^2 / (N * dv)`
       = `dv^2 / (N * dv) * |rfft(delta_F)[n]|^2`
       = `dv / N * |rfft(delta_F)[n]|^2`

(The rfft is normalised by numpy as a sum, not dividing by N, so we divide explicitly.)

## Averaging over sightlines

P1D is averaged over all sightlines (691200 per file):

    <P1D(k)> = (1 / N_skewers) × sum_i P1D_i(k)

## k range and units

- Pixel width: dv ≈ 10 km/s (exact value from Header)
- N_pix = 1556 pixels per sightline
- k_min = 1 / (1556 × 10) ≈ 6.4 × 10^-5 s/km
- k_max = 1 / (2 × 10) = 0.05 s/km  (Nyquist)
- k units: s/km

Default output bins: 35 bins matching the emulator kf grid from 0.00108 to 0.01951 s/km.

## P1D variants

| Variant | Description |
|---------|-------------|
| `all` | All sightlines, no masking |
| `no_DLA` | DLA pixels replaced by mean flux |
| `no_subDLA` | subDLA pixels replaced |
| `no_LLS` | LLS pixels replaced |
| `no_HCD` | All of LLS+subDLA+DLA replaced |

## Ratios

    ratio_noDLA_all    = P1D(no_DLA) / P1D(all)
    ratio_noHCD_all    = P1D(no_HCD) / P1D(all)
    ratio_DLA_contribution = 1 - ratio_noDLA_all

## Parseval check

By Parseval's theorem:

    sum_n P1D(k_n) × dk ≈ <delta_F^2>

where the sum is over positive k modes. This is verified in tests.
