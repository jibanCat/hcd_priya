# Where these three files come from

Measurements of the column-density distribution f(N_HI) below the DLA threshold, drawn as points and as a region in
the figure `docs/figures/cddf_vs_ho21.png` (`scripts/figures/fig_cddf_vs_ho21.py`), beside the DLA measurement of
Ho, Bird and Garnett (2021) (`../ho21/PROVENANCE.md`).

- **Where the files originate:** https://github.com/sbird/DLA_data, the collection of absorber measurements maintained
  by Simeon Bird, in its top folder. `zafar_2013.txt` was added there in commit
  `90104dd49364f0478e7dccd52675a41794af4033` ("Add data from Zafar Peroux 2013", 2013-10-07); `summary.dat` and
  `prochaska_lls.dat` in commit `3498694e59701b72c738e5eef950c62b73683c1e` ("Initial commit of all data and functions
  to plot it", 2013-09-13). No later commit of that repository changes any of the three.
- **Where they were fetched from:** the fork https://github.com/jibanCat/DLA_data, at commit
  `daa595c692254c93b86009318a6e2b029db67112` (branch `master`, 2025-02-27), the commit from which the `ho21` tables
  were fetched. The three files are the same in both repositories: each has the same Git blob id in the fork at that
  commit and in the origin's `master` (`147c616`), checked through the GitHub API on 2026-11-17.
- **Files, copied unchanged** (sha256, and the Git blob id in both repositories):

  | File | What | sha256 | Git blob |
  |---|---|---|---|
  | `zafar_2013.txt` | f(N_HI) of subDLAs, z 1.5 to 5 | `f592ecfd74bb423eae15d9bf40c7aca3e3da61ee7bbc1dab9510e6edca4767bd` | `4bfd121b92581ba481652ed74b2162487194eb01` |
  | `summary.dat` | f(N_HI) for log10 N_HI 19.0 to 20.3, mean z 3.1 | `19a3e15287d9f4d053fba2686ad985a51fba47c1a5b8963bee175b55f5bea40d` | `95c43f54b2b4118a81f2d13643df6b35408df407` |
  | `prochaska_lls.dat` | the region of f(N_HI) allowed for LLS, z 3.7 | `e0dd4bad98e5396acad183fb88b29562adcd83fd3d9b80c39a431a1c172c319e` | `3d5902c49f76448df44746f06f0222d7d04e7cd2` |

- **Citations** (title, authors, journal fields and DOI read from the Crossref records of the DOIs on 2026-11-17):
  - `zafar_2013.txt`: Zafar, T., Peroux, C., Popping, A., et al. 2013, "The ESO UVES advanced data products quasar
    sample. II. Cosmological evolution of the neutral gas mass density", A&A 556, A141,
    [doi:10.1051/0004-6361/201321154](https://doi.org/10.1051/0004-6361/201321154) (arXiv:1307.0602, as the
    collection's reader names it).
  - `summary.dat`: O'Meara, J. M., Prochaska, J. X., Burles, S., et al. 2007, "The Keck+Magellan Survey for Lyman
    Limit Absorption. I. The Frequency Distribution of Super Lyman Limit Systems", ApJ 656, 666,
    [doi:10.1086/510711](https://doi.org/10.1086/510711).
  - `prochaska_lls.dat`: Prochaska, J. X., O'Meara, J. M., Worseck, G. 2010, "A Definitive Survey for Lyman Limit
    Systems at z ~ 3.5 with the Sloan Digital Sky Survey", ApJ 718, 392,
    [doi:10.1088/0004-637X/718/1/392](https://doi.org/10.1088/0004-637X/718/1/392).
- **Licence:** neither repository has a licence file (no licence file in the origin, and no licence file in the
  fork). The files are redistributed here by decision of the project lead (2026-11-16).
- **Not used from the collection:** `peroux05_z3.dat` (it overlaps Zafar et al. 2013) and `kim13.txt` (column
  densities below the LLS class). The collection holds no dN/dX of subDLAs or LLS, so the dN/dX figure keeps its DLA
  points only.
- **How each file is read,** against the collection's own reader in `dla_data.py`, identical in the origin and in the
  fork at the commits above (the readers were read, not run; a test restates each mapping in numpy and compares):

  | File | The collection's reader | Here | Same numbers? |
  |---|---|---|---|
  | `zafar_2013.txt` | `zafar_data`: one row per bin; N_HI = 10**column 0, with x errors 10**(column 0 +- column 1) about it; f = 10**column 2; the upper y error from 10**(column 2 + column 3), the lower from 10**(column 2 - column 4); grey circles | `read_zafar13`: log10 N_HI = column 0 with a half-width of column 1 in log10 N_HI; f and its limits 10**(column 2 - column 4) and 10**(column 2 + column 3); open squares | yes: the same points and bars, the x axis in log10 N_HI |
  | `summary.dat` | `omeara_data`: a header line, then one row per bin, all log10; `format_error_bars` takes 10** of every column and turns the bin edges (columns 2, 3) and the limits of f (columns 4, 5) into distances from the point (columns 0, 1); black squares | `read_omeara07`: the same columns, kept as log10 N_HI, its two bin edges, f and its two limits; filled diamonds | yes |
  | `prochaska_lls.dat` | `prochaska_10_data`: six corners, log10 N_HI and log10 f; `ax.fill` of 10** of both, grey | `read_prochaska10`: the corners, log10 N_HI and f; a filled region | yes: the same polygon |
  | `moment=True` (all three) | multiplies f by N_HI | not used | the figure draws f(N_HI), as `moment=False` |

- **Which panel draws what:** the subDLA points of Zafar et al. (z 1.5 to 5) in every panel; the points of O'Meara et
  al. (mean z 3.1) and the region of Prochaska, O'Meara and Worseck (z 3.7) in the panel 3 < z < 4 only. The model
  is drawn at one snapshot per panel; these measurements are at other redshifts or over wider ones, and the
  simulations and the surveys define and find absorbers differently.
