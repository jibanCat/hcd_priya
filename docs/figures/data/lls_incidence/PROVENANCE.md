# Where this file comes from

Published measurements of the incidence of Lyman limit systems (LLS) per unit redshift, l(z) = dN/dz, for the
figure `docs/figures/dndx_vs_ho21.png` (`scripts/figures/fig_dndx_vs_ho21.py`), which draws them as dN/dX beside the
model's lines. The table was built on 2026-11-17 from the papers' own tables and stated values, by decision of the
project lead (2026-11-17: a traceable table, with its provenance and how to reproduce it). No value in the file is
converted or rescaled.

| File | What | sha256 |
|---|---|---|
| `lls_lz_literature.dat` | l(z) of LLS with tau_912 >= 2 in 18 published redshift bins from z = 1.947 to 5.40, six papers | `c7fd68c9effd2f0151fa5293ca57214a106c31e1e4611d63a6442b2b2ff47ae1` |

The sha256 of the 18 data rows alone (`grep -v '^#' lls_lz_literature.dat | sha256sum`) is
`630b928881e8dec1bc764a2af3da618cec3c2cfbc6f60279d11ae4867172d3bc`.

## What the rows count

- **Every row counts all absorbers above the Lyman-limit threshold, tau_912 >= 2, that is N_HI >= 10^17.5 cm^-2.**
  The count is cumulative: subDLAs and DLAs above the threshold are included. Each paper's own definition is quoted
  below. The model's LLS class is the binned range 10^17.2 <= N_HI < 10^19.0 (`hcd_analysis/catalog.py`), which is
  not the same set of absorbers; a model curve that counts the same absorbers as these rows counts every absorber
  above 10^17.5. The figure draws that curve (the dashed line), from the model's column-density distribution.
- **Every row is l(z)**, a number per unit redshift. It needs no cosmology. The figure turns it into dN/dX in the
  cosmology of the model line: l(X) = l(z) Delta z / Delta X over the row's bin [z_lo, z_hi], with Delta z = z_hi -
  z_lo and Delta X = integral of (1+z)^2 / sqrt(Omega_m (1+z)^3 + 1 - Omega_m) dz in the flat cosmology of the
  line (`figures_common.absorption_distance`). The errors take the same factor. This weights every redshift of the
  bin equally; a survey's own path g(z) weights them unequally. For the six SDSS bins of Prochaska et al. (2010),
  whose g(z) can be computed from the catalogue named below, the two differ by 0.03 to 0.22 per cent.
- The `lX` and `omega_m` columns exist for rows published only per unit X; this file has none (every paper gives
  l(z)), so `omega_m` is `nan` in every row.

## What the figure draws

The file keeps all 18 rows. The figure draws five samples (`LLS_DRAWN` in `scripts/figures/fig_dndx_vs_ho21.py`),
12 rows, so that no absorber is counted twice:

| Drawn | Rows | Marker |
|---|---|---|
| `prochaska10_t4` | 6 | triangle up |
| `omeara13_t5_comb` | 1 | square |
| `fumagalli13_nc` | 1 | diamond |
| `crighton19_corr` | 3 | triangle down |
| `fumagalli20_dr16` | 1 | hollow circle |

- **Left out, and why** (see "Rows that share data" below): `omeara13_t5_wfc3`, `omeara13_t5_acs` and
  `ribaudo11_t5_r2` are inside `omeara13_t5_comb`; the three rows of `ribaudo11_t7_rp10` count the SDSS-DR7 absorbers
  of `prochaska10_t4`.
- **The errors of `prochaska10_t4` and `omeara13_t5_comb` are drawn as a Poisson interval:** the larger error
  upward, that is with `err_lo` and `err_hi` swapped (the rule for the errors, under Format). For the combined
  O'Meara row the two printed errors are equal, so nothing changes. The file keeps the print. The script refuses a
  row of these two samples whose printed plus error is the larger one.
- **`fumagalli20_dr16` has its own marker,** a hollow circle: its absorbers were classified as tau >~ 2, and the
  paper says its catalogue may include some just below tau = 2 (the source entry below).
- **Each row is drawn at the centre of its redshift bin,** with a horizontal bar over the bin, as l(X) in the
  cosmology of the model line (above); the factors Delta z / Delta X of the drawn rows are in the side file
  `docs/figures/dndx_vs_ho21.json`.

## Format

Lines starting with `#` are the header. Each data row has eight tab-separated fields:

| Column | Meaning |
|---|---|
| `z_lo`, `z_hi` | the redshift bin, as printed |
| `value` | the published l(z), as printed |
| `err_lo`, `err_hi` | the published lower and upper 1 sigma errors, as printed (rule below) |
| `quantity` | `lz` (per unit redshift) or `lX` (per unit absorption distance) |
| `omega_m` | the paper's flat Omega_m for an `lX` row; `nan` for an `lz` row |
| `source` | the key of the source (below) |

Read the numbers with `np.loadtxt(path, usecols=(0, 1, 2, 3, 4, 6), ndmin=2)` and all fields as text with
`np.genfromtxt(path, dtype=str)`.

**The rule for the errors.** For a value printed as v^{+a}_{-b} (or "v +a -b"), `err_lo` = b and `err_hi` = a; for
"v +- c", both are c; for separate "-1 sigma" and "+1 sigma" columns, those two. The errors are copied as printed.
In the rows `prochaska10_t4` and `omeara13_t5_wfc3` / `omeara13_t5_acs`, the error printed with the plus sign is the
smaller one, the reverse of a Poisson interval of a count: for `prochaska10_t4` the Poisson interval of the counts
(the check of these rows below) gives lower errors 0.43, 0.41, 0.23, 0.27, 0.52, 0.68 and upper errors 0.60, 0.49,
0.27, 0.31, 0.63, 0.84, which are the printed '+' and '-' values; for the two O'Meara rows the Poisson interval of the
printed m and Delta z gives 0.23 and 0.29, 0.39 and 0.61. The file keeps the print. A figure that draws the Poisson
interval swaps `err_lo` and `err_hi` in these rows; one that draws the print does not. The figure draws the Poisson
interval ("What the figure draws").

## The sources

The arXiv e-prints are gzipped tar files fetched from `https://arxiv.org/e-print/<id><version>`; for each, the
sha256 of the tar file and of the file read inside it. Each version named is the latest on arXiv.

### `prochaska10_t4` (6 rows): Prochaska, O'Meara and Worseck (2010)

- **Paper:** Prochaska, J. X., O'Meara, J. M., Worseck, G. 2010, "A Definitive Survey for Lyman Limit Systems at
  z ~ 3.5 with the Sloan Digital Sky Survey", ApJ 718, 392,
  [doi:10.1088/0004-637X/718/1/392](https://doi.org/10.1088/0004-637X/718/1/392), arXiv:0912.0292 (version 1, the
  only version, the one submitted to the journal).
- **Source:** the journal's Table 4, "LLS Statistics", on the article page
  https://iopscience.iop.org/article/10.1088/0004-637X/718/1/392, read from a copy saved on 2026-11-17 (sha256
  `08c0571653e11e484bedd37b3c2932da00285324023b4c4b39bf724264341b3d`; caption at HTML line 534, the six rows under
  "Redshift intervals" at lines 583 to 628, the notes at lines 632 to 636). The page is generated on each request,
  so another copy has another sha256; its values are those below. The page also offers the table as ASCII
  (https://iopscience.iop.org/0004-637X/718/1/392/suppdata/apj339002t4_ascii.txt).
- **Definition** (Table 4, note b): "Number of tau_912 >= 2 LLS discovered in the survey path"; note d: "Line
  densities of LLS with tau >= 2 per redshift dz or absorption length dX". The arXiv source says the same
  (`prpt_dr7lls.tex`, line 1559: `Number of $\mtll \ge 2$ LLS discovered in the survey path`, with `\mtll` defined
  as `\tau_{\rm 912}` at line 53).
- **Cosmology** (Table 4 notes): Omega_Lambda = 0.7, Omega_m = 0.3, H_0 = 72 km/s/Mpc; used only for the Delta X and
  l(X) columns, which are not copied.
- **Rows copied:** the column l_tau>=2(z) for the S/N^A_912 = 2 threshold (the table's default: "Unless specified
  otherwise, all quantities refer to the S/N^A_912 = 2 threshold"), the six redshift intervals [3.30,3.40], [3.40,3.50],
  [3.50,3.65], [3.65,3.90], [3.90,4.10], [4.10,4.40]: 1.31 +0.43 -0.60, 2.18 +0.41 -0.49, 1.60 +0.23 -0.27, 2.04
  +0.27 -0.31, 2.84 +0.52 -0.63, 3.22 +0.68 -0.84.
- **The arXiv version differs:** e-print 0912.0292v1 (tar sha256
  `587134286398edce7179f34daec980d79bdcb448b46e2f38f8cfc4d9fbc03a64`), `prpt_dr7lls.tex` (sha256
  `f236288f014668c136b1f4a5b4e5d4c19b741f0538d3c090edeabffe2bd574b9`), table `tab:llssumm`, lines 1550 to 1555,
  prints 2.17, 1.59, 2.05 and 2.89 in the bins [3.40,3.50], [3.50,3.65], [3.65,3.90] and [3.90,4.10] (with m 57 and
  30 in the last two). The journal values are used; the survey's catalogue reproduces them (the check below).

### `omeara13_t5_wfc3`, `omeara13_t5_acs`, `omeara13_t5_comb` (1 row each): O'Meara et al. (2013)

- **Paper:** O'Meara, J. M., Prochaska, J. X., Worseck, G., Chen, H.-W., Madau, P. 2013, "The HST/ACS+WFC3 Survey for
  Lyman Limit Systems. II. Science", ApJ 765, 137,
  [doi:10.1088/0004-637X/765/2/137](https://doi.org/10.1088/0004-637X/765/2/137), arXiv:1204.3093 (version 2).
- **Source:** e-print 1204.3093v2 (tar sha256 `a87cc2d71132f7596a4fa67d4eb8af9fc9b5ed287f392830a1cc2d68f6e15078`),
  `tab5.tex` (sha256 `61c43be3e2cca635296e809423304ad1f2960841f55a6d15a2e4e68cb9081fe5`), Table 5 "Incidence of LLS"
  (label `tab:incidence`): the cut heads `$\mtll \ge 2$` at lines 17 (WFC3/UVIS-G280), 20 (HST/ACS) and 23
  ("Combined (with Ribaudo et al. 2011)"), and their [2.00,2.60] rows at lines 19, 22 and 25. The journal's Table 5
  (article page, copy of 2026-11-17, sha256 `50bc7758b4c31c386038adbf5d269c05bb8883f52f7fe54a9c9dd29c40eb6f13`)
  prints the same values.
- **Definition** (`ms.tex`, sha256 `693b673c4b17ae11df27e341c8bf594fa7467a7a48b5eacd40f2a239aabd0bb1`, lines 476 to
  478): "an optical depth at the Lyman limit tau_912 ~ N_HI/10^17.19 cm^-2. In the following, we refer to systems with
  tau_912 >= 2 as Lyman limit systems (LLSs)" (`\mtll` is `\tau^{\rm LL}_{\rm 912}`, line 88).
- **Cosmology** (`ms.tex` lines 367 and 368; Table 5 note): H_0 = 72 km/s/Mpc, Omega_m = 0.26, Omega_Lambda = 0.74;
  used only for Delta X and l(X), not copied.
- **Rows copied:** the l(z) column of the three [2.00,2.60] rows: WFC3 0.96^{+0.23}_{-0.29}, ACS 0.91^{+0.39}_{-0.61},
  Combined 0.92^{+0.18}_{-0.18}. The Combined row is WFC3 + ACS + the [1.947,2.594] bin of Ribaudo et al. (2011)
  (the table's note; m 17 + 5 + 10 = 32 and Delta z 17.7 + 5.5 + 11.56 = 34.8 as printed).

### `ribaudo11_t5_r2` (1 row), `ribaudo11_t7_rp10` (3 rows): Ribaudo, Lehner and Howk (2011)

- **Paper:** Ribaudo, J., Lehner, N., Howk, J. C. 2011, "A Hubble Space Telescope Study of Lyman Limit Systems:
  Census and Evolution", ApJ 736, 42, [doi:10.1088/0004-637X/736/1/42](https://doi.org/10.1088/0004-637X/736/1/42), arXiv:1105.0659 (version 1, the
  only version).
- **Source:** e-print 1105.0659v1 (tar sha256 `f03d34881cf84d910ced29373a87ee47d86bd80554235513ac7aa5a016aaceb8`),
  `aph_man.tex` (sha256 `9b6e97146e990d02fb1b62566cf77a8f97666b1bbdf3a0bc55b429a3ae62300e`): Table 5 "Redshift
  Density of LLSs for R1 and R2" (label `tab:stats`, caption line 2368), cut `$\tau_{\rm LLS} \geq 2$` at line 2384,
  row [1.947,2.594] at line 2390; Table 7 "Redshift Density of LLSs for RP10" (label `tab:stats2`, caption line
  2434), rows [3.500,3.663], [3.663,3.925], [3.925,4.907] at lines 2445 to 2447. Table 7's row [1.947,2.594] (line
  2444) repeats line 2390 and is not repeated in the file. The table numbers are those of the source's comment lines.
- **Definition** (line 236): "The second sample, R2, defines a standard LLS as an absorber where tau_LLS >= 2, i.e.,
  N_HI >= 10^17.5 cm^-2." RP10 (lines 603 and 604): "We refer to the combined R2 and [Prochaska et al. 2010] samples
  as the RP10 sample. This combined sample of HST and SDSS observations contains 685 QSOs and 206 LLSs with tau_LLS
  >= 2."
- **Cosmology** (lines 191 and 192): H_0 = 70 km/s/Mpc, Omega_m = 0.3, Omega_Lambda = 0.7; not needed for l(z).
- **Rows copied:** the l(z) column: 0.86 +- 0.30; 1.80 +- 0.26, 1.99 +- 0.29, 2.57 +- 0.44.

### `fumagalli13_nc` (1 row): Fumagalli et al. (2013)

- **Paper:** Fumagalli, M., O'Meara, J. M., Prochaska, J. X., Worseck, G. 2013, "Dissecting the properties of
  optically-thick hydrogen at the peak of cosmic star formation history", ApJ 775, 78,
  [doi:10.1088/0004-637X/775/1/78](https://doi.org/10.1088/0004-637X/775/1/78), arXiv:1308.1101 (version 2).
- **Source:** e-print 1308.1101v2 (tar sha256 `ea784801e41260a3a8f815107ab03df73d19d7619330e07e224b9c08690d93cf`),
  `llsz3.tex` (sha256 `9b04b5020ec0ada97dec15d8811deb5ee348c0ed1980e95d6792738492465c2d`), line 651: "In our
  statistical sample between 2.6 <= z <= 3, we find l(z)_tau>=2 = 1.21 +- 0.28 for the non-color selected sample";
  the redshift range as written at line 604, "2.6 <= z <= 3.0".
- **Definition** (lines 645 to 649): "we define the incidence of tau >= 2 LLSs as the total number of systems detected
  within the useful redshift range divided by the integral of g(z)"; lines 103 and 104: "N_HI >= 10^17.5 cm^-2, the
  column density at which the optical depth at the Lyman limit exceeds tau >= 2".
- **Cosmology** (lines 131 and 132): Omega_Lambda = 0.728, Omega_m = 0.272, H_0 = 70.4 km/s/Mpc; not needed for l(z).
- **Row copied:** 1.21 +- 0.28, the non-colour-selected sample, the one the paper analyses (lines 485 to 487: "we
  focus our statistical analysis on the non-color selected samples, providing results from the color selected
  sample for comparison only").

### `crighton19_corr` (3 rows): Crighton et al. (2019)

- **Paper:** Crighton, N. H. M., Prochaska, J. X., Murphy, M. T., O'Meara, J. M., Worseck, G. 2019, "Imprints of the
  first billion years: Lyman limit systems at z ~ 5", MNRAS 482, 1456, [doi:10.1093/mnras/sty2762](https://doi.org/10.1093/mnras/sty2762),
  arXiv:1810.10040 (version 1, the only version).
- **Source:** e-print 1810.10040v1 (tar sha256 `41a901015825da2055face22f3d56530a6fd683a1f65f92d145123f42e520dfb`),
  `tab_lzlX.tex` (sha256 `59d384d25dfdcbc0ee7032c2de122b8721001fb98a9f519529fdcea660512a20`), table `tab:lzlX`, "Raw
  and corrected LLS incidence rates in the GGG spectra based on one author's LLS identifications (NC)": column heads
  at line 6, rows 3.75--4.40, 4.40--4.70, 4.70--5.40 at lines 9 to 11.
- **Definition** (`ggg_lls.tex`, sha256 `c8b88ee984f35368eda6b861889c360622362a6e44245c003d8915ee0e55e524`, lines 334
  and 335): "we decided to limit our analysis to LLSs with tau_LL >= 2 in this paper"; lines 339 and 340: "An LLS with
  tau_LL >= 2, corresponds to N_HI >= 10^17.5 cm^-2".
- **Cosmology** (line 252): H_0 = 70 km/s/Mpc, Omega_m = 0.3, Omega_Lambda = 0.7; not needed for l(z).
- **Rows copied:** the corrected l(z) (column 6, after the correction for survey biases found with mock spectra) and
  its "-1 sigma" and "+1 sigma" columns (7 and 8): 2.21 -0.34 +0.40, 2.20 -0.39 +0.47, 2.95 -0.49 +0.58. The raw
  l(z) (column 5) and the l(X) columns are not copied.

### `fumagalli20_dr16` (1 row): Fumagalli, Fotopoulou and Thomson (2020)

- **Paper:** Fumagalli, M., Fotopoulou, S., Thomson, L. 2020, "Detecting neutral hydrogen at z >~ 3 in large
  spectroscopic surveys of quasars", MNRAS 498, 1951,
  [doi:10.1093/mnras/staa2388](https://doi.org/10.1093/mnras/staa2388), arXiv:2009.03322 (version 1, the only
  version).
- **Source:** e-print 2009.03322v1 (tar sha256 `3a9382f945d54fdbf7ce174e83c2f6dd150be18e7c3ca9354972526db531c847`),
  `mllls.tex` (sha256 `806b6d43760631b056e40b415dac55cc9db63356957a6c4142d978974b52f1f4`), line 407 (section 5): "we
  compute l(z) as the ratio of the number of LLSs in the range z=[3.3,3.6] and the total redshift path ..., finding
  l(z) = 2.32 +- 0.08. The error bar is based on Poisson statistics."
- **Definition:** line 283: "we limit our classification only to tau >~ 2 LLSs, which are strong enough to produce a
  clear and (nearly) saturated absorption at the Lyman limit"; line 403: "the redshift of the first tau >= 2 LLS
  identified"; line 92 (abstract): "LLSs with N_HI >~ 10^17.5 cm^-2". The sample is SDSS DR16 quasars with z > 3.6
  and S/N > 5.5 at 1150 A, without proximate LLSs (lines 399 and 400). The paper notes (line 413) that its value may
  be raised by "the inclusion of LLSs just below the tau = 2 boundary in our catalogue" and calls it about 30 per
  cent above the trend of the earlier measurements (line 412).
- **Cosmology:** not stated for l(z), and not needed.
- **Row copied:** 2.32 +- 0.08.

## How to reproduce the table

1. Fetch the six e-prints and check them (any `curl` or a browser; arXiv may answer HTTP 429 to fast repeated
   requests: wait and repeat):

   ```
   for id in 0912.0292v1 1204.3093v2 1105.0659v1 1308.1101v2 1810.10040v1 2009.03322v1; do
       curl -sS -L -o $id.tar.gz https://arxiv.org/e-print/$id; sleep 5
       mkdir -p x_$id && tar -xzf $id.tar.gz -C x_$id
   done
   sha256sum *.tar.gz x_0912.0292v1/prpt_dr7lls.tex x_1204.3093v2/tab5.tex x_1105.0659v1/aph_man.tex \
             x_1308.1101v2/llsz3.tex x_1810.10040v1/tab_lzlX.tex x_2009.03322v1/mllls.tex
   ```

   The digests must equal those given above. (On 2026-11-17 the tar files fetched this way were byte-identical to
   copies fetched earlier the same day.)
2. Print the lines and copy the values, in the printed digits:

   ```
   sed -n '17,25p' x_1204.3093v2/tab5.tex              # rows 19, 22, 25: column 6 is l(z)
   sed -n '2384,2390p;2434p;2444,2447p' x_1105.0659v1/aph_man.tex   # column 6 is l(z)
   sed -n '604p;651,652p' x_1308.1101v2/llsz3.tex
   sed -n '6p;9,11p' x_1810.10040v1/tab_lzlX.tex       # columns 1, 6, 7, 8
   sed -n '407p' x_2009.03322v1/mllls.tex
   ```

   For `prochaska10_t4`, open Table 4 of the journal article (link above) and copy the first l_tau>=2(z) column
   (S/N_912 = 2) of the six "Redshift intervals" rows. The next section recomputes these rows from the survey's
   catalogue, as a check of the published table.
3. Write one tab-separated row per bin in the order of the file: z_lo, z_hi, value, err_lo, err_hi, `lz`, `nan`,
   source key, with the error rule above. Then `grep -v '^#' lls_lz_literature.dat | sha256sum` on the rebuilt file
   (rows only) must give `630b928881e8dec1bc764a2af3da618cec3c2cfbc6f60279d11ae4867172d3bc`; with the header block
   copied from this file, the whole file gives `c7fd68c9effd2f0151fa5293ca57214a106c31e1e4611d63a6442b2b2ff47ae1`.

## A check of the `prochaska10_t4` rows

The values of `prochaska10_t4` are those of the journal's Table 4; no value in the file comes from anywhere else.
As a check of that published table, its rows were recomputed from the catalogue of the survey. pyigm
(https://github.com/pyigm/pyigm, commit `fef47716146bdafbf073c43603a5e24511ff3e38`; its `setup.py` declares a
3-clause BSD licence) ships the SDSS-DR7 LLS survey of Prochaska et al. (2010) and the code that counts it. Its
counting, restated in numpy, reproduces the journal's Table 4.

- **Files:** `pyigm/data/LLS/SDSS/lls_dr7_stat_LLS.fits.gz` (1935 LLS; sha256
  `d286af52e69e226ee83bde8efbfe73f6aa620d6e9c662167cee5b030815b3818`) and
  `pyigm/data/LLS/SDSS/lls_dr7_qsos_sn2050.fits.gz` (3759 quasars; sha256
  `48ed9d7540389354ecd60a8b85927e3d0a388945bf4ecdecee9bf63858731982`), from
  `https://raw.githubusercontent.com/pyigm/pyigm/fef47716146bdafbf073c43603a5e24511ff3e38/<path>`.
- **Selection** (`LLSSurvey.load_SDSS_DR7(sample='stat')`, `pyigm/surveys/llssurvey.py` lines 400 to 472, and
  `lls_stat`, line 574 on): a quasar contributes path only if z_em >= 3.6 (otherwise its ZT2 is set to 99.99); the
  path starts at Z_START = max(ZT2, ZLLS, z_em - 0.4), quasars with Z_START <= 0 or FLG_QSO != 0 are dropped, and it
  ends at Z_END, 3000 km/s below z_em (relativistic velocity). An LLS is statistical when its N_HI > 10^17.5, it
  matches its quasar (nearest on the sky, |delta z_em| < 0.03), and Z_START - 1e-4 < z_abs < Z_END. This gives 191
  LLS (pyigm's test `pyigm/surveys/tests/test_llssurvey.py` line 47 asserts 191).
- **Counting** (`IGMSurvey.binned_loz(zbins, NHI_mnx=(17.49, 23.))`, `pyigm/surveys/igmsurvey.py` lines 249 to 304,
  with `calculate_gz`, line 447): m = the statistical LLS with zbins[k] <= z_abs < zbins[k+1]; Delta z = the sum of
  g(z) dz over the same range, g(z) on a grid of step 1e-4 in z; l(z) = m / Delta z; errors from the Poisson
  interval of m (`astropy.stats.poisson_conf_interval(m, 'frequentist-confidence')`). `binned_lox` gives Delta X the
  same way for flat Omega_m = 0.3.
- **Result** (bins of Table 4): m = 9, 28, 46, 56, 29, 22 (journal: the same); Delta z = 6.846, 12.839, 28.792,
  27.511, 10.222, 6.837 (journal 6.8, 12.8, 28.8, 27.5, 10.2, 6.8); Delta X = 25.724, 48.842, 111.058, 108.355,
  41.282, 28.336 (journal 25.7, 48.8, 111.1, 108.3, 41.3, 28.3); l(z) = 1.3146, 2.1808, 1.5976, 2.0355, 2.8372,
  3.2180, which round to the journal's values; Poisson lower and upper errors 0.4300/0.6004, 0.4097/0.4949,
  0.2347/0.2720, 0.2712/0.3100, 0.5238/0.6307, 0.6808/0.8427, which round to the journal's '+' and '-' values in that
  order. The full range [3.30,4.40] gives m = 190, Delta z = 93.05, Delta X = 363.60 (journal 190, 93.1, 363.6). The
  arXiv v1 counts (57 and 30 in [3.65,3.90] and [3.90,4.10]) are not reproduced: the catalogue is the journal's.

## Rows that share data

- `omeara13_t5_comb` contains `omeara13_t5_wfc3`, `omeara13_t5_acs` and `ribaudo11_t5_r2`.
- `ribaudo11_t7_rp10` (z > 3.5) counts LLS of the SDSS-DR7 survey of `prochaska10_t4`: its R2 sample ends at z =
  2.594. Counted in pyigm's catalogue, its bins hold 56, 54 and 44 LLS against the printed 56, 55 and 45.
- `fumagalli20_dr16` uses SDSS DR16 spectra, which include DR7 quasars of `prochaska10_t4` (344 quasars in common,
  `mllls.tex` line 371), with its own classification.
- `crighton19_corr` overlaps `prochaska10_t4` in redshift (3.75 to 4.40) but uses other spectra (the Giant Gemini GMOS
  survey of z ~ 5 quasars).

## Not included, and why

- **Songaila and Cowie (2010)**, ApJ 721, 1448, doi:10.1088/0004-637X/721/2/1448, arXiv:1007.3262v2 (tar sha256
  `72c7f85dcfa52a0a2673f06b21299b68952a7320fe97a8570e5d63d2ac71bf26`; `songaila_REV_LANL.tex`, sha256
  `cb64d64c66aa8c01997297aca518bc03c27e4e9ef9c169e7f15729fcee39b1f3`): its LLS have **tau > 1**, not tau >= 2. Line
  324: "below which the spectrum dropped by more than a factor of 2.7 (tau = 1)"; line 389: "strong LLS (tau > 1)";
  line 634: "dn is the number of tau > 1 LLS in the interval dX". Its Table 3 (lines 415 to 433) counts a larger set
  of absorbers (N_HI >= 10^17.2) than the rows here.
- **Other rows of the tables used:** the S/N_912 = 3 column and the full-range row [3.30,4.40] of Prochaska et al.
  (2010) Table 4 (another threshold, and the union of the six bins); the tau_912 >= 0.5 and >= 1 rows and the
  [1.20,2.00] rows of O'Meara et al. (2013) Table 5; the rows below z = 1.947 and the tau_LLS >= 1 cut of Ribaudo et
  al. (2011); the colour-selected value of Fumagalli et al. (2013) (1.86 +- 0.46, "for comparison only"); the raw
  l(z) of Crighton et al. (2019), and its full-survey l(z) of 2.6 +- 0.4.
- **The l(X) columns** of every table: l(z) needs no cosmology, and the figure converts it in the cosmology of its
  model line.

## Journal versions

The values of Prochaska et al. (2010) are the journal's (above). The journal Table 5 of O'Meara et al. (2013) was
compared with the arXiv source and prints the same values. For Ribaudo et al. (2011), Fumagalli et al. (2013),
Crighton et al. (2019) and Fumagalli et al. (2020) the arXiv versions named above (each the latest) were read; their
journal pages were not compared.

## How the file was checked

Every row was read three times: once by hand into the file, once more by hand in each source's printed notation, and
once by a program that parses the cited lines of the sources after checking their sha256; the three agree in all 18
rows. The rows of Prochaska et al. (2010) were also recomputed from pyigm's catalogue as above (numpy only, without
pyigm), and the definition sentences quoted above were checked in the sources by the same program.
