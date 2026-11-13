# Figure data

The data files that the figure scripts in `scripts/figures/` read, each with its origin. `SHA256SUMS` lists every
data file; a script refuses a file whose digest differs.

| File | What | Origin |
|---|---|---|
| `ho21/cddf_z225.txt`, `ho21/cddf_z253.txt`, `ho21/cddf_z34.txt` | The CDDF of DLAs measured by Ho, Bird and Garnett (2021), in three redshift bins | [`ho21/PROVENANCE.md`](ho21/PROVENANCE.md) |
| `ho21/dndx_all.txt` | The incidence dN/dX of DLAs measured by Ho, Bird and Garnett (2021), in eighteen redshift bins | [`ho21/PROVENANCE.md`](ho21/PROVENANCE.md) |
| `class_ratio_z3p0_lf60.csv` | The class power ratio P_class / P_clean of the 60 low-resolution PRIYA training simulations at z = 3, with the filtered troughs restored and as filtered: the mean and the standard deviation over the simulations | Made for this figure from the project's training simulations; its header says how. The ratio is formed in each simulation at fixed mode number and then averaged mode by mode; the `k` listed, where the points are plotted, is the mean over the simulations of the wavenumber of that mode (the boxes differ slightly in length in km/s; the relative standard deviation of k over the simulations is about 3.9e-2 at most). The public predictor cannot produce this table: it returns the filtered class spectra |
| `priya_lf_design_60.csv` | The nine parameters of the 60 low-resolution simulations of the PRIYA suite, one row per simulation: the design points the emulator is trained on. The figure scripts evaluate the model at each of them to draw the bands | The design of the PRIYA suite (Bird et al. 2023, JCAP 2023 (10) 037, [arXiv:2306.05471](https://arxiv.org/abs/2306.05471)) together with the simulations added later in a widened parameter range (the last 12 rows), as stored in the design file of the emulator directory this project trains from; the header gives that file's sha256. Committed as figure data with the approval of the project's PI (2026-11-13) |
