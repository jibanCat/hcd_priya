# Where these four files come from

Measurements of damped Lyman-alpha absorbers by Ho, Bird and Garnett (2021): the column-density distribution in
three redshift bins, the measured points of the figure `docs/figures/cddf_vs_ho21.png`
(`scripts/figures/fig_cddf_vs_ho21.py`); and the incidence dN/dX against redshift, the measured points of the
figure `docs/figures/dndx_vs_ho21.png` (`scripts/figures/fig_dndx_vs_ho21.py`).

- **Where the tables originate:** https://github.com/sbird/DLA_data, folder `ho21/`, the collection of DLA
  measurements maintained by Simeon Bird. They were added there in commit
  `147c616ba4f5779bea0ef3e23255f4a9c2021cea` ("Add data from Ho+ 21", 2021-05-06).
- **Where they were fetched from:** the fork https://github.com/jibanCat/DLA_data, folder `ho21/`, at commit
  `daa595c692254c93b86009318a6e2b029db67112` (branch `master`, 2025-02-27), on 2026-11-13. The four files are the
  same in both repositories: each has the same Git blob id at the two commits (checked through the GitHub API on
  2026-11-13). The fork differs from its origin in `dla_data.py` only, and not in the function `ho21_cddf`.
- **Files, copied unchanged** (sha256, and the Git blob id in both repositories):

  | File | Redshift bin | sha256 | Git blob |
  |---|---|---|---|
  | `cddf_z225.txt` | 2 to 2.5 | `feaffb7b88215e767ca53a41fb3f00b65c90aaf27745c28720b32976a9168a63` | `9dc846086352df9f50ebcd67849d8922fbb0083b` |
  | `cddf_z253.txt` | 2.5 to 3 | `62e93a1e1977b3d78b9f56475bd4390c1eee7eaf371986ca5ece2db338bb7cc7` | `24a84a088f8df754e544a7f299305c4b69137ec5` |
  | `cddf_z34.txt` | 3 to 4 | `f21dca06cd5974a08de13632ef74109f5b78afc2921f86ec9b3c6cf9f838ad3a` | `ed221507c84fd453a9ac6a23d0d6fb026690658b` |
  | `dndx_all.txt` | 2 to 5, eighteen bins | `fb4d86151b34240902dbd6e63914e7c98dbb367fb6880a0e6871c849345ca81c` | `da849aa301b957be2ac55fa94f2806027b76a547` |

- **Citation:** Ho, M.-F., Bird, S., Garnett, R. 2021, "Damped Lyman-alpha absorbers from Sloan Digital Sky Survey
  DR16Q with Gaussian processes", MNRAS, [doi:10.1093/mnras/stab2169](https://doi.org/10.1093/mnras/stab2169),
  [arXiv:2103.10964](https://arxiv.org/abs/2103.10964).
- **Licence:** neither repository has a licence file (no licence file in the origin, and no licence file in the
  fork). The files are redistributed here by decision of the project lead, who is the first author of the paper and
  owns the fork.
- **What the rows of the three CDDF files are,** read from the function `ho21_cddf` of `dla_data.py`, which is
  identical in the origin and in the fork at the commits above (the function was read, not run): each file has six rows of thirty numbers. Row 0: log10 of N_HI (N_HI in
  cm^-2). Row 1: f(N_HI). Rows 2 and 3: the lower and the upper limit of the 68 per cent interval. Rows 4 and 5: the
  lower and the upper limit of the 95 per cent interval.
- **Two readings that are inferred from the numbers, not stated by the reader:** that row 0 is the bin centre (the
  values run from 20.05 to 22.95 in steps of 0.1), and that a bin with f = 0 holds no detection (such bins still
  carry a non-zero upper limit; the figure does not draw them).
- **This project's reader against `ho21_cddf`, step by step** (`read_ho21_cddf` in
  `scripts/figures/fig_cddf_vs_ho21.py`; a test pins the mapping):

  | Step | `ho21_cddf` | Here | Same numbers? |
  |---|---|---|---|
  | Reading | `np.loadtxt(file).T`: one row per bin, six columns | `np.loadtxt(file)`: six rows, one column per bin | yes: column i there is row i here |
  | Column density | `NHI = 10**data[:, 0]`, drawn on a logarithmic axis | row 0, log10 of N_HI, drawn on a linear axis | yes: the same positions |
  | f(N_HI) | `data[:, 1]` | row 1 | yes |
  | Limits, 68 per cent | with `onesigma=True`: lower `data[:, 2]`, upper `data[:, 3]` | rows 2 and 3 | yes |
  | Limits, 95 per cent | the default (`onesigma=False`): `data[:, 4]`, `data[:, 5]` | not drawn | the figure shows the 68 per cent interval |
  | Error bars | `yerr = [cddf - lower, upper - cddf]` | `yerr = [f - lower, upper - f]` | yes |
  | `moment=True` | multiplies f and the limits by N_HI | not used | the figure draws f(N_HI), as `moment=False` |
  | Units or other conversions | none | none | yes |
  | Bins with f = 0 | passed to the plot like every other bin | not drawn | a difference in drawing: see below |
  | File for 3 <= z < 4 | `ho21/cddf_z253.txt` | `ho21/cddf_z34.txt` | a difference in data: see below |

- **Two differences, both stated in the figure's caption:**
  - **Bins in which the measured f is zero** (none in the 2 to 2.5 table, four in the 2.5 to 3 table, six in the 3
    to 4 table). `ho21_cddf` hands them to the plot, where the point is off a logarithmic axis and the upper limit
    can show as a bar. The figure here does not draw these bins.
  - **The file for 3 <= z < 4.** `ho21_cddf` opens `ho21/cddf_z253.txt` for both 2.5 <= z < 3 and 3 <= z < 4, in
    the origin and in the fork. The figure here takes `ho21/cddf_z34.txt` for the bin 3 to 4, the file named for
    that range; the two files hold different numbers. The paper's first author confirmed on 2026-11-14 that
    `cddf_z34.txt` is the table of the bin 3 to 4.
- **What the rows of `dndx_all.txt` are.** Neither repository has a reader for this file. The first author of
  the paper, who owns the fork and maintains this project, confirmed the rows on 2026-11-13: six rows
  of eighteen numbers. Row 0: the redshift of the bin centre. Row 1: dN/dX of DLAs. Rows 2 and 3: the lower and the
  upper limit of the 68 per cent interval. Rows 4 and 5: the lower and the upper limit of the 95 per cent interval.
  The figure shows the 68 per cent interval.
- **Not copied:** the CDDF file for 4 < z < 5 (the figure leaves that bin out), the CDDF file for all redshifts, and
  the table of Omega_DLA.
