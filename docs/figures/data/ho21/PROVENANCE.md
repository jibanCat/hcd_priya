# Where these four files come from

Measurements of damped Lyman-alpha absorbers by Ho, Bird and Garnett (2021): the column-density distribution in
three redshift bins, the measured points of the figure `docs/figures/cddf_vs_ho21.png`
(`scripts/figures/fig_cddf_vs_ho21.py`); and the incidence dN/dX against redshift, the measured points of the
figure `docs/figures/dndx_vs_ho21.png` (`scripts/figures/fig_dndx_vs_ho21.py`).

- **Source repository:** https://github.com/jibanCat/DLA_data, folder `ho21/`.
- **Commit fetched:** `daa595c692254c93b86009318a6e2b029db67112` (branch `master`, 2025-02-27), on 2026-11-13.
- **Files, copied unchanged** (sha256, and the Git blob id in the source repository):

  | File | Redshift bin | sha256 | Git blob |
  |---|---|---|---|
  | `cddf_z225.txt` | 2 to 2.5 | `feaffb7b88215e767ca53a41fb3f00b65c90aaf27745c28720b32976a9168a63` | `9dc846086352df9f50ebcd67849d8922fbb0083b` |
  | `cddf_z253.txt` | 2.5 to 3 | `62e93a1e1977b3d78b9f56475bd4390c1eee7eaf371986ca5ece2db338bb7cc7` | `24a84a088f8df754e544a7f299305c4b69137ec5` |
  | `cddf_z34.txt` | 3 to 4 | `f21dca06cd5974a08de13632ef74109f5b78afc2921f86ec9b3c6cf9f838ad3a` | `ed221507c84fd453a9ac6a23d0d6fb026690658b` |
  | `dndx_all.txt` | 2 to 5, eighteen bins | `fb4d86151b34240902dbd6e63914e7c98dbb367fb6880a0e6871c849345ca81c` | `da849aa301b957be2ac55fa94f2806027b76a547` |

- **Citation:** Ho, M.-F., Bird, S., Garnett, R. 2021, "Damped Lyman-alpha absorbers from Sloan Digital Sky Survey
  DR16Q with Gaussian processes", MNRAS, [doi:10.1093/mnras/stab2169](https://doi.org/10.1093/mnras/stab2169),
  [arXiv:2103.10964](https://arxiv.org/abs/2103.10964).
- **Licence:** the source repository has no licence file. The files are used here at the direction of the
  repository's owner, who is the first author of the paper and the maintainer of this project.
- **What the rows of the three CDDF files are,** read from the function `ho21_cddf` of `dla_data.py` in the source repository at that
  commit (the function was read, not run): each file has six rows of thirty numbers. Row 0: log10 of N_HI (N_HI in
  cm^-2). Row 1: f(N_HI). Rows 2 and 3: the lower and the upper limit of the 68 per cent interval. Rows 4 and 5: the
  lower and the upper limit of the 95 per cent interval.
- **Two readings that are inferred from the numbers, not stated by the reader:** that row 0 is the bin centre (the
  values run from 20.05 to 22.95 in steps of 0.1), and that a bin with f = 0 holds no detection (such bins still
  carry a non-zero upper limit; the figure does not draw them).
- **One slip in the source repository's reader, not followed here:** `ho21_cddf` reads `cddf_z253.txt` for both
  2.5 <= z < 3 and 3 <= z < 4. Each file is taken here by the redshift range in its name.
- **What the rows of `dndx_all.txt` are.** The source repository has no reader for this file. The first author of
  the paper, who owns the source repository and maintains this project, confirmed the rows on 2026-11-13: six rows
  of eighteen numbers. Row 0: the redshift of the bin centre. Row 1: dN/dX of DLAs. Rows 2 and 3: the lower and the
  upper limit of the 68 per cent interval. Rows 4 and 5: the lower and the upper limit of the 95 per cent interval.
  The figure shows the 68 per cent interval.
- **Not copied:** the CDDF file for 4 < z < 5 (the figure leaves that bin out), the CDDF file for all redshifts, and
  the table of Omega_DLA.
