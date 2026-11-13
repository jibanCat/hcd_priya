# The class spectra against the model of Rogers et al. (2018)

The predictor returns the P1D of four classes of sightlines: clean, LLS, subDLA and DLA. The same decomposition was
introduced by Rogers et al. (2018), who fitted a model of each class's P1D relative to the clean forest on the
Illustris simulation. This page records one comparison of the PRIYA simulations behind hcd_priya with that model.

**In one sentence:** with the filtered troughs restored, the per-class ratios of the PRIYA simulations agree with the
per-class model of Rogers et al. (2018, their Eq. 6) over most of k < 0.03 s/km: to 0.4% to 1.3% for LLS, 0.6% to
2.2% for subDLA, and 1.0% to 8.9% for DLA depending on how the two DLA templates are mixed (the median over k; the
table below); the amplitudes at the lowest wavenumbers differ.

## What was compared

- **Simulations only.** No survey data enters, and no emulator prediction.
- **The simulation side.** The ratio `R = P_class / P_clean` from the 60 low-resolution training simulations, taken
  on the mean over the simulations, at the mean-flux rung nearest a tau0 amplitude of 1, at redshifts 2.6 to 4.2 and
  wavenumbers k < 0.03 s/km.
- **With the filtered troughs restored.** The class spectra that the predictor returns are PRIYA-filtered: pixels
  with an optical depth above 1e6 are filled. That removes the cores of the strongest absorbers, and the predictor
  states it as its two omissions (`pred.labels["omissions"]`): the power of the subDLA cores and the DLA
  damping-wing core are not added back. For this comparison both were added back.
- **The model side.** The per-class ratio of Rogers et al. (2018), their Eq. 6, including its constant c(z), as coded
  in `hcd_analysis.hcd_template.class_ratio`:

  ```text
  P_class / P_forest (k, z) = f_z(z) * (a(z) * exp(b(z) * k) - 1)**-2 + c(z)
  ```

- **The DLA class.** Rogers et al. split the DLAs into small and large. The simulations here have one DLA class, so
  it is compared with a mix of the two templates, once with the shares of Rogers et al. (0.585 and 0.415) and once
  with the simulations' own shares.

## The numbers

The median over k of `|Rogers / R - 1|`, as a range over the redshifts. `s` is the amplitude of the k-dependent term
that fits the simulations best, where 1 is the amplitude of Rogers et al.

| Class | Against Eq. 6, `c(z) + T` | Against `1 + T` | `s`, troughs restored | `s`, filtered |
|---|---|---|---|---|
| LLS (restoring the troughs changes nothing) | 0.4% to 1.3% | 3.0% to 3.6% | 1.25 to 1.54 | the same |
| subDLA | 0.6% to 2.2% | 13% to 16% | 1.5 to 1.9 (1.36 to 1.67 without the lowest mode) | 0.58 to 0.61 |
| DLA, the shares of Rogers et al. | 2.8% to 8.9% | 54% to 75% | 0.70 to 0.99 (0.95 to 1.03 without the lowest mode) | 0.02 to 0.09 |
| DLA, the simulations' own shares | 1.0% to 3.3% | 54% to 75% | 0.86 to 1.17 (0.94 to 1.03 without the lowest mode) | 0.02 to 0.11 |

Three things to read with the table:

- **The constant matters.** `1 + T` is the fitting form of Rogers et al. (their Eq. 8) with its free normalisation
  set to 1. It is not their prediction for one class: that is Eq. 6, `c(z) + T`. The constant c is 0.95 to 0.97 for
  LLS, 0.87 for subDLA, 0.67 to 0.70 for small DLAs and 0.36 to 0.43 for large DLAs, so comparing a class ratio with
  `1 + T` is off by 13% to 75% for the subDLA and DLA classes.
- **The median hides the troughs.** Against Eq. 6 the filtered spectra also give small medians (subDLA 0.3% to 4%,
  DLA 2% to 8%). The median is set by the plateau above k of about 0.005 s/km, where the troughs add 0.4% to 8%.
  Restoring them matters at the lowest modes: at z = 3 the DLA ratio at the lowest mode is 0.84 filtered and 11.6
  restored. The filter removes the low-k rise of the subDLA and DLA classes, which is what the column `s` shows.
- **What does not agree at that level: the low-k amplitude.** The LLS and subDLA classes of PRIYA rise more steeply
  towards low k than the template (`s` of 1.3 to 1.9); the large-DLA class rises less (`s` of 0.47 to 0.89). The
  cause is not established. Two things are known about the two sides at low k:
  - **The template.** It was fitted on a smaller box than PRIYA's. Below roughly the lowest modes of that box it is an
    extrapolation and is not reliable there. This is an order-of-magnitude argument from the box size, not a
    measurement of the template's error.
  - **The simulations.** Bird et al. (2023) estimate the cosmic variance of the PRIYA flux power spectrum from two
    simulations that differ only in their initial seed (their section 3.1 and Figure 5): about 2 per cent at z < 3.6,
    and 2 to 5 per cent for k < 3e-3 s/km at z >= 4. Those numbers are for the total flux power, not for the class
    ratios compared here, whose own variance was not measured.

## The comparison at z = 3, drawn

![The class power ratio at z = 3 of the PRIYA training simulations with the troughs restored, and Eq. 6 of Rogers et al. (2018)](figures/class_ratio_vs_rogers18.png)

Points: the simulations with the troughs restored, the mean and the standard deviation over the 60 simulations, at
z = 3.0. For this figure the DLA sightlines of the simulations are split into small and large at log10 N_HI = 21, as
in Rogers et al.; the table above compares one DLA class with a mix of the two templates. Line: Eq. 6 of Rogers et
al. (2018) from `hcd_template.class_ratio`, evaluated at z = 3.0; their snapshot is at z = 3.01. Shaded: below 2 pi / 8000 km/s = 7.85e-4 s/km, the
lowest mode of their box, where the template is evaluated by extrapolation. The figure holds no emulator prediction.
Its script and data: [the figures page](model_figures.md).

## What "restoring the troughs" means

- The restored term is the unfiltered class power minus the filtered class power, at the same optical-depth scaling
  and with one global mean flux.
- It is exactly zero for the clean class, and exactly zero for the LLS class at redshifts 2.6 to 4.2. It is
  non-zero for the subDLA and DLA classes in every case.
- The filter removes the core of an absorber and the part of its wings above the fill threshold; the restored term
  brings back both.
- The historical forward model of this repository adds back the DLA term only and leaves the subDLA class
  filtered. This comparison restores both.

## The template as coded

- `hcd_analysis/hcd_template.py` holds the coefficients a0, a1, b0, b1, c0, c1, the pivot redshift 2 and the redshift
  slope -3.55 exactly as in Table 2 of Rogers et al.
- The wavenumber convention (angular, in s/km) and the mean flux (one mean over all spectra) are those of the paper.

## Differences of definition that remain

- **Simulation and box.** Illustris against PRIYA, with their different incidence of absorbers. The sightlines differ
  in length by a factor of 0.60: the lowest mode of PRIYA lies below the lowest mode of the box of Rogers et al.,
  where the template is an extrapolation.
- **Mean flux.** PRIYA rescales the optical depth so that the filtered sample has the target mean flux; the
  unfiltered sample is then 0.9% to 1.8% darker.
- **Class assignment.** The class boundaries are the same; the assignment is not. Rogers et al. use the largest
  column density integrated over four pixels, PRIYA the strongest catalogued absorber of the sightline. The share of
  small DLAs among the DLAs is 0.69 to 0.73 in PRIYA and 0.55 to 0.60 in Rogers et al.
- **Pixels.** Pixel size and smoothing differ. Whether that cancels in the ratio was not tested.
- **The template itself** is a six-parameter fit at five redshifts, with no stated accuracy.

## What this comparison is not

- **Not a test of the emulator.** The emulator predicts the filtered class spectra. Its predictions with the troughs
  restored were not tested.
- **Not a statement about any survey data.**
- **Not an explanation of the low-k difference.** The amplitudes at the lowest wavenumbers differ and the cause is
  not established: the simulation, the box and the sightline length differ, and the template is extrapolated below
  the lowest mode of its own box (see "What does not agree" above).

## What you can run

Only the model side runs from the public package. The simulation side needs the training cache of the simulations,
which is not distributed.

```python
import numpy as np
from hcd_analysis.hcd_template import class_ratio

k = np.array([0.001, 0.003, 0.01, 0.03])        # angular wavenumbers in s/km
ratio = class_ratio(k, 3.0)                     # Rogers et al. 2018, Eq. 6, at z = 3
print("classes of the model:", sorted(ratio))
print("shape of one class ratio:", ratio["LLS"].shape)
# far above the k range of the comparison the ratio tends to the constant c(z) of each class
high_k = class_ratio(np.array([1.0]), 3.0)
print("the constant c(z = 3) of each class:", {name: round(float(v[0]), 2) for name, v in high_k.items()})
# the single DLA class of the simulations is compared with a mix of the two DLA templates
dla = 0.585 * ratio["Small-DLA"] + 0.415 * ratio["Large-DLA"]
print("the mixed DLA ratio falls with k over this range:", bool(np.all(np.diff(dla) < 0)))
```

It prints:

```text
classes of the model: ['LLS', 'Large-DLA', 'Small-DLA', 'Sub-DLA']
shape of one class ratio: (4,)
the constant c(z = 3) of each class: {'LLS': 0.97, 'Sub-DLA': 0.87, 'Small-DLA': 0.68, 'Large-DLA': 0.38}
the mixed DLA ratio falls with k over this range: True
```

The four constants are those of the published coefficients; they are not simulation results.

## References

Rogers, K. K., Bird, S., Peiris, H. V., Pontzen, A., Font-Ribera, A., Leistedt, B. 2018, "Simulating the effect of
high column density absorbers on the one-dimensional Lyman-alpha forest flux power spectrum", MNRAS, 474 (3),
3032-3042, [doi:10.1093/mnras/stx2942](https://doi.org/10.1093/mnras/stx2942),
[arXiv:1706.08532](https://arxiv.org/abs/1706.08532).

Bird, S., Fernandez, M., Ho, M.-F., Qezlou, M., Monadi, R., Ni, Y., Chen, N., Croft, R., Di Matteo, T. 2023, "PRIYA: a
new suite of Lyman-alpha forest simulations for cosmology", JCAP 2023 (10) 037,
[doi:10.1088/1475-7516/2023/10/037](https://doi.org/10.1088/1475-7516/2023/10/037),
[arXiv:2306.05471](https://arxiv.org/abs/2306.05471).
