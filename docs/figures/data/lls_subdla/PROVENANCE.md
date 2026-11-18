# Where these three files come from

Measurements of the column-density distribution f(N_HI) below the DLA threshold, drawn as points and as a region in
the figure `docs/figures/cddf_vs_ho21.png` (`scripts/figures/fig_cddf_vs_ho21.py`), beside the DLA measurement of
Ho, Bird and Garnett (2021) (`../ho21/PROVENANCE.md`). One file is copied from a collection of absorber measurements;
the other two are built from the published tables of their papers.

| File | What | sha256 |
|---|---|---|
| `zafar_2013.txt` | f(N_HI) of subDLAs, z 1.51 to 5.00 | `f592ecfd74bb423eae15d9bf40c7aca3e3da61ee7bbc1dab9510e6edca4767bd` |
| `omeara2007_fn_all19p0.dat` | f(N_HI) for log10 N_HI 19.0 to 20.3, mean z 3.10 (sample ALL-19.0) | `695fd5e611368249b17b8de0ccf42aab8ee5c74c68f0750f7997c6c3d81f70bc` |
| `prochaska2010_lls_region.dat` | the region of f(N_HI) allowed for LLS, log10 N_HI 17.5 to 19.0, z ~ 3.7 | `9afad095522ac3d25ea1b615df91b806758ab50b42268dd6fea37412d689b39b` |

## `zafar_2013.txt`: from the collection of absorber measurements

- **Where it originates:** https://github.com/sbird/DLA_data, the collection of absorber measurements maintained by
  Simeon Bird, in its top folder, added in commit `90104dd49364f0478e7dccd52675a41794af4033` ("Add data from Zafar
  Peroux 2013", 2013-10-07); no later commit changes it.
- **Where it was fetched from:** the fork https://github.com/jibanCat/DLA_data, at commit
  `daa595c692254c93b86009318a6e2b029db67112` (branch `master`, 2025-02-27), the commit from which the `ho21` tables
  were fetched. The file has the same Git blob id, `4bfd121b92581ba481652ed74b2162487194eb01`, in the fork at that
  commit and in the origin's `master` (`147c616`), checked through the GitHub API on 2026-11-17. Copied unchanged.
- **Citation** (from the Crossref record of the DOI, read on 2026-11-17): Zafar, T., Peroux, C., Popping, A., et al.
  2013, "The ESO UVES advanced data products quasar sample. II. Cosmological evolution of the neutral gas mass
  density", A&A 556, A141, [doi:10.1051/0004-6361/201321154](https://doi.org/10.1051/0004-6361/201321154)
  (arXiv:1307.0602).
- **Compared with the paper** (2026-11-17): the arXiv e-print 1307.0602v2 (the latest version; the gzipped tar
  fetched from https://arxiv.org/e-print/1307.0602v2 has sha256
  `5e0a4999653979340a5d29664ebd22008247240f3f5d7c7b36ab0022d9f691dc`; the file read, `uvespaperII.tex`, sha256
  `ad3866f8002f9826702477775e01cb4faf7d0553a6450833b083b411ff79ba35`), the table `fnresults`, "Summary of the column
  density distribution of the combined sub-DLA sample", lines 267 to 272. The five rows of the file equal it: log f
  -20.43, -20.75, -21.15, -21.30, -21.61; upper errors (column 3) 0.09, 0.09, 0.09, 0.09, 0.11; lower errors (column 4)
  0.10, 0.10, 0.13, 0.12, 0.16, as printed v^{+a}_{-b}. Column 0 is the middle of the table's log N_HI range (19.00 to
  19.30, 19.30 to 19.60, 19.60 to 19.90, 19.90 to 20.10, 20.10 to 20.30) and column 1 its half width; the table's
  mean log N_HI (19.13, 19.42, 19.75, 20.06, 20.06) is not used. The journal version was not compared.
- **Licence:** neither repository has a licence file (no licence file in the origin, and no licence file in the
  fork). The file is redistributed here by decision of the project lead (2026-11-16).

## `omeara2007_fn_all19p0.dat`: built from the table of O'Meara et al. (2007)

- **Paper:** O'Meara, J. M., Prochaska, J. X., Burles, S., Prochter, G., Bernstein, R. A., Burgess, K. M. 2007, "The
  Keck + Magellan Survey for Lyman Limit Absorption. I. The Frequency Distribution of Super Lyman Limit Systems", ApJ
  656, 666, [doi:10.1086/510711](https://doi.org/10.1086/510711), arXiv:astro-ph/0610726 (version 2).
- **Source:** the arXiv e-print astro-ph/0610726v2 (https://arxiv.org/e-print/astro-ph/0610726v2), file
  `prpt_paper1.tex`, sha256 `5dd7e2e9891b2d5c16ffda8620ddeae74c5916a8c005fb303c4999d11ed84bb8`, read on 2026-11-17:
  the table labelled `tab_fn` (lines 750 to 771; Table 5 of the journal article, p. 676, which holds the same
  values), row "ALL-19.0" (line 767), its four columns of log f_HI(N, X) for the bins [19.0, 19.3), [19.3, 19.6),
  [19.6, 19.9) and [19.9, 20.3) (column heads, lines 758 to 761). ALL-19.0 is the sample of MIKE and UVES spectra
  with log10 N_HI from 19.0 to 20.3 (Delta X = 329.8, mean z = 3.10, 60 systems), the one with mean z 3.1 and the bin
  from 19.0.
- **Columns** (all log10; the six columns of the collection's reader `omeara_data`): x, the centre of the bin (a
  convention: the table gives bins; the paper's Fig. 8 plots this sample at the bin centres); y, the table's log f;
  xmin and xmax, the bin edges; ymin and ymax, y minus the table's lower error and y plus its upper error (in dex).
  The values are copied, not rescaled. The file's `#` lines repeat this.

## `prochaska2010_lls_region.dat`: built from the table of Prochaska, O'Meara and Worseck (2010)

- **Paper:** Prochaska, J. X., O'Meara, J. M., Worseck, G. 2010, "A Definitive Survey for Lyman Limit Systems at
  z ~ 3.5 with the Sloan Digital Sky Survey", ApJ 718, 392,
  [doi:10.1088/0004-637X/718/1/392](https://doi.org/10.1088/0004-637X/718/1/392), arXiv:0912.0292 (version 1, the
  only version, the one submitted to the journal).
- **Source:** the arXiv e-print 0912.0292v1 (https://arxiv.org/e-print/0912.0292v1), file `prpt_dr7lls.tex`, sha256
  `f236288f014668c136b1f4a5b4e5d4c19b741f0538d3c090edeabffe2bd574b9`, read on 2026-11-17: the table labelled `tab:fn`
  ("SUMMARY OF f(N) RESULTS", lines 2100 to 2137), cut "Conservative Range of Allowed Values" (line 2110), its four
  extreme rows (lines 2114, 2115, 2118 and 2119: log f at N_HI = 10^19 of -20.25 or -19.85, and log k_LLS of -4.1,
  4.7, -18.0 and -4.6); the power law f = k_LLS N^beta_LLS for N_HI from 10^17.5 to 10^19 is that of the section
  "f(N) at z ~ 3.7" (lines 1921 to 1946).
- **Construction:** each extreme row is the line log f = log f19 + beta (log N - 19), with beta = (log f19 - log
  k_LLS) / 19 (the table's one-decimal beta column agrees after rounding); the region is between the lowest and the
  highest of the four lines at each log N, from 17.5 to 19.0; the file holds its six corners (log10 N_HI, log10 f).
  The paper's own Fig. 10, read as vector data from the e-print, draws the same region to 0.002 dex. A test
  recomputes the corners from the four rows.
- **Not compared:** the table of the journal article. The arXiv version is the submitted one; the journal PDF was not
  read.

## The files these two replace

Earlier versions of this pull request drew `summary.dat` and `prochaska_lls.dat` of the same collection (added there
in commit `3498694e59701b72c738e5eef950c62b73683c1e`, "Initial commit of all data and functions to plot it",
2013-09-13). Their values could not be traced to a published source:

- `summary.dat` differs from the published table (row ALL-19.0 above, the same in both arXiv versions and in the
  journal) by +0.02 to +0.07 dex in log f (+0.066, +0.054, +0.040 and +0.022 dex from the first bin to the last).
  The difference is not constant, so it is no cosmology or Delta X rescaling, and no public version of the paper,
  and none of the f(N) tables in the authors' public code repositories that were searched, holds these numbers.
- `prochaska_lls.dat` appears digitised from Fig. 10 of the paper: its corners lie within 0.02 dex of those computed
  from the table.

By decision of the project lead (2026-11-17), the figure now draws files built from the papers' own tables.

## The cosmology of each measurement, and how the figure converts it

f(N_HI, X) is a count per unit absorption distance X, and each survey computed its Delta X in its own cosmology. All
three are flat with Omega_m = 0.3, as each paper states: Zafar et al. (2013), section 1, last paragraph (arXiv
`uvespaperII.tex` line 85, the file named above); O'Meara et al. (2007), end of section 1 (line 215); Prochaska, O'Meara and Worseck
(2010), section 1, last paragraph (line 329). The figure draws them, like the DLA points, in the cosmology of the
model line: flat, with Omega_m = omegamh2 / hub^2 of the parameters it is drawn at (0.143 / 0.70^2 = 0.2918). Each
measurement's values and limits (its error bars, and the region's corners) are multiplied by the same factor,

    factor = Delta X(own Omega_m) / Delta X(line Omega_m),
    Delta X = integral from z1 to z2 of (1 + z)^2 / sqrt(Omega_m (1 + z)^3 + 1 - Omega_m) dz

(`figures_common.to_line_cosmology`), over the redshift range of its sample:

| File | Redshift range (source) | Factor |
|---|---|---|
| `zafar_2013.txt` | 1.51 to 5.00: the combined sub-DLA sample of the f(N) figure (caption, line 242) and of the table `nzresults` (line 173) | 0.9870 (-0.0057 dex) |
| `omeara2007_fn_all19p0.dat` | 1.68 to 4.47: "the redshift range of the sample" (line 87) | 0.9871 (-0.0057 dex) |
| `prochaska2010_lls_region.dat` | 3.3 to 4.4: the survey's row [3.30, 4.40] of the table `tab:llssumm`, "LLS STATISTICS" (line 1548) | 0.9867 (-0.0058 dex) |

The data files stay as published; the factors are applied when the figure is drawn and written to its side file
`docs/figures/cddf_vs_ho21.json`. The band of the figure (the model at the 60 design points) is not converted: each
design point is per unit X of its own cosmology (Omega_m from 0.254 to 0.339), so only the line is in the
measurements' cosmology. Put in the cosmology of the band's two extremes instead, the points would move by -6.0 to
-6.6 per cent (Omega_m 0.254) and +6.9 to +7.6 per cent (Omega_m 0.339).

## How each file is read

The collection's own readers in `dla_data.py` (identical in the origin and in the fork at the commits above) were
read, not run; a test restates each mapping in numpy and compares. The two built files keep the collection's formats.

| File | The collection's reader | Here | Same numbers? |
|---|---|---|---|
| `zafar_2013.txt` | `zafar_data`: one row per bin; N_HI = 10**column 0, with x errors 10**(column 0 +- column 1) about it; f = 10**column 2; the upper y error from 10**(column 2 + column 3), the lower from 10**(column 2 - column 4); grey circles | `read_zafar13`: log10 N_HI = column 0 with a half-width of column 1 in log10 N_HI; f and its limits 10**(column 2 - column 4) and 10**(column 2 + column 3); open squares | yes: the same points and bars, the x axis in log10 N_HI |
| `omeara2007_fn_all19p0.dat` | `omeara_data`, on a file of this format: a header, then one row per bin, all log10; `format_error_bars` takes 10** of every column and turns the bin edges (columns 2, 3) and the limits of f (columns 4, 5) into distances from the point (columns 0, 1); black squares | `read_omeara07`: the same columns, kept as log10 N_HI, its two bin edges, f and its two limits; filled diamonds | yes |
| `prochaska2010_lls_region.dat` | `prochaska_10_data`, on a file of this format: six corners, log10 N_HI and log10 f; `ax.fill` of 10** of both, grey | `read_prochaska10`: the corners, log10 N_HI and f; a filled region | yes: the same polygon |
| `moment=True` (all three) | multiplies f by N_HI | not used | the figure draws f(N_HI), as `moment=False` |

## Which panel draws what

The subDLA points of Zafar et al. (z 1.51 to 5.00) in every panel; the points of O'Meara et al. (mean z 3.1) and the
region of Prochaska, O'Meara and Worseck (z ~ 3.7) in the panel 3 < z < 4 only. The model is drawn at one snapshot
per panel; these measurements are at other redshifts or over wider ones, and the simulations and the surveys define
and find absorbers differently. The dN/dX figure draws no file of this folder: its LLS points come from
`../lls_incidence/` (`PROVENANCE.md` there).
