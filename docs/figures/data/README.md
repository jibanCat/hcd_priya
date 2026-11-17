# Figure data

The data files that the figure scripts in `scripts/figures/` read, each with its origin. `SHA256SUMS` lists every
data file; a script refuses a file whose digest differs.

## Where the measurements are

Every published measurement the figures draw is a plain text file under `docs/figures/data/`, readable with numpy
alone. Each folder has a `PROVENANCE.md` that names the paper, the table and lines the values come from, and how the
file was made. The values are as published: f(N_HI, X) and dN/dX are per unit absorption distance X in each paper's
own cosmology, and l(z) per unit redshift; the figure scripts convert them to the cosmology of the model line.

```text
docs/figures/data/
├── README.md                          this page
├── SHA256SUMS                         the sha256 of every data file below
├── ho21/                              DLAs in SDSS DR16Q, Ho, Bird and Garnett (2021); ho21/PROVENANCE.md
│   ├── cddf_z225.txt                  CDDF at 2 < z < 2.5  ┐ 6 rows of 30 bins: log10 N_HI, f(N_HI, X),
│   ├── cddf_z253.txt                  CDDF at 2.5 < z < 3  │ the 68 per cent lower and upper limits,
│   ├── cddf_z34.txt                   CDDF at 3 < z < 4    ┘ the 95 per cent lower and upper limits
│   └── dndx_all.txt                   dN/dX of DLAs: 6 rows of 18 bins: z, dN/dX, 68 and 95 per cent limits
├── lls_subdla/                        the CDDF below the DLA threshold; lls_subdla/PROVENANCE.md
│   ├── zafar_2013.txt                 subDLAs, Zafar et al. (2013): one row per bin: log10 N_HI, half width,
│   │                                  log10 f, upper and lower error (dex)
│   ├── omeara2007_fn_all19p0.dat      O'Meara et al. (2007), log10 N_HI 19 to 20.3: one row per bin: centre,
│   │                                  log10 f, bin edges, log10 f at the lower and the upper error
│   └── prochaska2010_lls_region.dat   Prochaska, O'Meara and Worseck (2010): the six corners (log10 N_HI,
│                                      log10 f) of the region allowed for LLS at z 3.7
├── lls_incidence/                     LLS with tau_912 >= 2; lls_incidence/PROVENANCE.md
│   └── lls_lz_literature.dat          l(z) of six papers, one row per published redshift bin: z_lo, z_hi,
│                                      l(z), lower and upper error, quantity, Omega_m, source key
├── priya_lf_design_60.csv             not a measurement: the 60 design points of the bands
└── class_ratio_ranges_z3p0_lf60.csv   not a measurement: class power ratios of the training simulations
```

Lines starting with `#` are comments (`np.loadtxt` skips them). The table below gives each file's origin.

| File | What | Origin |
|---|---|---|
| `ho21/cddf_z225.txt`, `ho21/cddf_z253.txt`, `ho21/cddf_z34.txt` | The CDDF of DLAs measured by Ho, Bird and Garnett (2021), in three redshift bins | The tables of https://github.com/sbird/DLA_data, fetched from its fork https://github.com/jibanCat/DLA_data: [`ho21/PROVENANCE.md`](ho21/PROVENANCE.md) |
| `ho21/dndx_all.txt` | The incidence dN/dX of DLAs measured by Ho, Bird and Garnett (2021), in eighteen redshift bins | The tables of https://github.com/sbird/DLA_data, fetched from its fork https://github.com/jibanCat/DLA_data: [`ho21/PROVENANCE.md`](ho21/PROVENANCE.md) |
| `lls_subdla/zafar_2013.txt`, `lls_subdla/omeara2007_fn_all19p0.dat`, `lls_subdla/prochaska2010_lls_region.dat` | The CDDF below the DLA threshold: of subDLAs (Zafar et al. 2013, z 1.51 to 5.00); for log10 N_HI 19.0 to 20.3 (O'Meara et al. 2007, mean z 3.1); and the region allowed for LLS (Prochaska, O'Meara and Worseck 2010, z 3.7) | The subDLA table is a file of https://github.com/sbird/DLA_data, fetched from its fork https://github.com/jibanCat/DLA_data; the other two are built from the papers' own published tables (the arXiv sources, line by line): [`lls_subdla/PROVENANCE.md`](lls_subdla/PROVENANCE.md) |
| `lls_incidence/lls_lz_literature.dat` | The incidence l(z) = dN/dz of Lyman limit systems with tau_912 >= 2 (every absorber with N_HI >= 10^17.5 cm^-2), 18 published redshift bins from z 1.95 to 5.40, from Prochaska, O'Meara and Worseck (2010), Ribaudo, Lehner and Howk (2011), O'Meara et al. (2013), Fumagalli et al. (2013), Crighton et al. (2019) and Fumagalli, Fotopoulou and Thomson (2020); the dN/dX figure draws 12 rows of five of them | Copied from the papers' own tables and text (the journal's Table 4 for Prochaska et al. 2010, the arXiv sources line by line for the others), not converted: [`lls_incidence/PROVENANCE.md`](lls_incidence/PROVENANCE.md), with the steps to rebuild the file |
| `class_ratio_ranges_z3p0_lf60.csv` | The class power ratio P_class / P_clean of the low-resolution PRIYA training simulations at z = 3 with the filtered troughs restored, as ranges per mode (minimum, 16th, 50th and 84th percentile, maximum) over two sets: the 60 simulations at the mean-flux rung nearest 1, and the 60 simulations at all 20 rungs | Made for this figure from the project's training simulations; its header says how. The ratio is formed in each simulation at fixed mode number; the `k` listed is the mean over the simulations of the wavenumber of that mode. The public predictor cannot produce this table: it returns the filtered class spectra. Redistributed by decision of the project lead |
| `priya_lf_design_60.csv` | The nine parameters of the 60 low-resolution simulations the emulator is trained on, one row per simulation. The figure scripts evaluate the model at each of them to draw the bands | The 60 design points of the emulator files of Fernandez, Bird and Ho (2024): 48 are the design of Bird et al. (2023) and 12 were added by Fernandez et al., whose paper states "a total of 60 low fidelity (LF) and 3 high fidelity (HF) simulations". The points are the entry `sample_params` of `Emulator_Files/emulator_params.json` in the public repository https://github.com/mafern/InferenceLyaData, at commit `2c68e0749aaad38c752664eaf9d14882b764ba0b` (2024-05-25), sha256 `623aa4cad657d10ae9bb659f73d23ef001fc870fc63a947f635d6b912a1b740d`: the same bytes as the design file this project trains from, whose digest the table's header gives. The CSV holds the 60 rows in the file's order, with the nine names as header. That repository has no licence file; the table is used at the direction of the project lead. References: Bird et al. 2023, JCAP 2023 (10) 037 ([arXiv:2306.05471](https://arxiv.org/abs/2306.05471)); Fernandez, Bird and Ho 2024, JCAP 2024 (07) 029 ([arXiv:2309.03943](https://arxiv.org/abs/2309.03943)). Redistributed by decision of the project lead |
