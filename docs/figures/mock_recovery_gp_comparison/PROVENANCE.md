# The summary table of the comparison with the Gaussian-process emulator

`gp_comparison_posterior_summary.csv` holds 44 numbers: the posterior mean and standard deviation of eleven
parameters, for two fits of one mock. It is published so that the two numerical statements made about this
comparison (in the text beside Figure 3 of the JOSS paper on `hcd_priya`, and in the
[comment on pull request 26](https://github.com/jibanCat/hcd_priya/pull/26#issuecomment-6921376340) that first
showed it) can be recomputed from a public file. `recompute_from_summary.py` does that, from the table alone.

**It is a stored summary. It cannot be regenerated from public files, because the chains of the two fits are not
published.**

## What the two fits are

- **The mock** is the one of the example download of release 1.0.0 (`docs/likelihood.md`, "The example download"):
  the noiseless P1D of one PRIYA high-resolution simulation, named in the provenance of the mock file, at 11
  redshifts from 2.2 to 4.2 and 11 wavenumbers each (121 bins of the KODIAQ-SQUAD P1D). Both fits use this data
  vector with the data covariance alone: no emulator error is in the covariance.
- **`hcd_priya`** (columns `mean_hcd_priya`, `sd_hcd_priya`): the fold model of the example download, which was
  trained without the simulation of the mock, so for it the recovery is a held-out one. The fit is the one whose
  summary the example download ships as its stored reference (`reference.json`): the flat boxes and the two
  Gaussian terms of `examples/mock_recovery.py`, with the class spectra mixed by the model's own class fractions,
  so that the HCD content is not a free parameter. The chains were sampled with the project's private reference
  likelihood, which was compared with the likelihood of this package (`docs/likelihood.md`).
- **`gp`** (columns `mean_gp`, `sd_gp`): the Gaussian-process emulator of Ho et al. (2026), on the same bins and
  with the same data covariance alone, its Rogers factor off. **This emulator was trained with the simulation of the
  mock, so for it the fit is an in-sample one. The two fits are not like for like.**
- In both fits `hub` and `bhfeedback` are set by Gaussian terms centred away from the truth of this mock.

## How the table was made

- Four chains per fit. The first 30 per cent of the rows of each chain is dropped; the rest of the four chains is
  pooled.
- The mean is the mean over the pooled rows with the chains' sample weights; the standard deviation is the square
  root of the weighted mean of the squared deviations from that mean.
- Each number is written with six significant digits.
- The row `tau0` is the mean-flux amplitude, `tau0_amp` in `hcd_priya` and `tau0` in Ho et al. (2026). The rows are
  in the order of the stored reference of the example download.

## The chains behind the table

The table was made from the same eight chain files as Figure 3 of the JOSS paper; the figure's script pins their
sha256.

## What can be checked against public files

- The eleven values of `mean_hcd_priya` are the values of `mean` in the `reference.json` of the example download,
  to the six digits written here.
- The means of both fits and the ratios of the widths, rounded, are those of the table in the comment on pull
  request 26.
- The standard deviations of both fits are in no other public file: the stored reference holds intervals, not
  standard deviations.

## How to check this table

Three steps, with public files only. The first two need the example download of release 1.0.0, unpacked
(`docs/likelihood.md`, "The example download"); run the lines from the root of a clone, with the download beside
it or its path in place of `hcd_priya_example_download`.

```python
import csv
import json

reference = json.load(open("hcd_priya_example_download/reference.json"))
lines = open("docs/figures/mock_recovery_gp_comparison/gp_comparison_posterior_summary.csv")
table = list(csv.DictReader(line for line in lines if not line.startswith("#")))
for name, mean, (low, high), row in zip(
    reference["param_names"], reference["mean"], reference["interval68"], table
):
    half_over_sd = (high - low) / 2 / float(row["sd_hcd_priya"])
    print(f"{name:10s} {mean:.6g} {row['mean_hcd_priya']:>12s} {half_over_sd:.2f}")
```

1. **The means of `hcd_priya`.** The second and third columns printed are equal in every row: the `mean` of the
   stored reference, written with six significant digits, and `mean_hcd_priya` of the table, in the same order of
   parameters.
2. **The widths of `hcd_priya`.** The last column is half of the stored reference's central 68 per cent interval
   over `sd_hcd_priya`. It lies between 0.99 and 1.17 over the eleven parameters (it is 1.00 for a Gaussian), so
   the standard deviations of the table are consistent with the intervals that are already public.
3. **Both fits, rounded.** The table in the comment on pull request 26 gives, per parameter, the means of both fits,
   their 68 per cent intervals and the ratio of the widths, rounded. Each of its means and ratios is this table's
   value after rounding.

**What cannot be checked from public files:** the numbers of the Gaussian-process fit beyond the rounding of that
comment, and the standard deviations of both fits exactly. They rest on the chains, which are not published.

## The two numbers

`python docs/figures/mock_recovery_gp_comparison/recompute_from_summary.py` prints them:

- the largest difference of the two posterior means, in units of one fit's own standard deviation (for each
  parameter the larger of the two values, one per fit, is taken): 0.27, for `ns`;
- the ratio of the posterior widths, `hcd_priya` over the Gaussian-process emulator: between 1.00 and 1.09 over the
  eleven parameters, to two decimals.

## What the table does not give

- **No contours.** It holds no correlation between parameters, so the two-dimensional regions of a figure cannot be
  drawn from it.
- **No posterior predictive.** The percentiles of the posterior predictive of the `hcd_priya` fit are in the stored
  reference of the example download; nothing of that kind is here for the Gaussian-process fit.
- **No statement about any measurement.** It is one noiseless mock, with the data covariance alone.
