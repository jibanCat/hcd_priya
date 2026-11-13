# Figure data

The data files that the figure scripts in `scripts/figures/` read, each with its origin. `SHA256SUMS` lists every
data file; a script refuses a file whose digest differs.

| File | What | Origin |
|---|---|---|
| `ho21/cddf_z225.txt`, `ho21/cddf_z253.txt`, `ho21/cddf_z34.txt` | The CDDF of DLAs measured by Ho, Bird and Garnett (2021), in three redshift bins | [`ho21/PROVENANCE.md`](ho21/PROVENANCE.md) |
| `class_ratio_z3p0_lf60.csv` | The class power ratio P_class / P_clean of the 60 low-resolution PRIYA training simulations at z = 3, with the filtered troughs restored and as filtered: the mean and the standard deviation over the simulations | Made for this figure from the project's training simulations; its header says how. The ratio is formed in each simulation at fixed mode number and then averaged mode by mode; the `k` listed, where the points are plotted, is the mean over the simulations of the wavenumber of that mode (the boxes differ slightly in length in km/s; the relative standard deviation of k over the simulations is at most 3.9e-2). The public predictor cannot produce this table: it returns the filtered class spectra |
