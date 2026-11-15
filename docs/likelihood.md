# The Gaussian likelihood and the mock-recovery example

`hcd_analysis.likelihood` turns a loaded model and a P1D data set into a log likelihood. This page says what it
computes, what it leaves out, and how the example that comes with it is run. The reference of every function is on
the [API page](api.md).

**Nothing is fitted to a measurement by anything on this page.** The example recovers the parameters of a mock, a
data vector made from a simulation at a known point. No mock and no model is published yet. The example reads them
from the example download described below, which is not published yet, or from files you name.

## What is computed

For a parameter point (the nine PRIYA parameters, `tau0_amp` and `dtau0`):

1. The mean flux at each redshift of the data is PRIYA's two-parameter model (`predictor.tau_eff_priya`).
2. The model's class spectra at each redshift are mixed into one spectrum with the model's own class fractions,
   `add_hcd(pred, pred.w_c)`, or with amplitudes you fix. With the model's fractions the HCD content follows the
   nine parameters: it is not varied on its own, and it is not fixed at a truth either.
3. The mixed spectrum is brought to the wavenumbers of the data by linear interpolation in k, at the wavenumber of
   each bin. It is the model at a point, not an average over the window of a bin.
4. The result is multiplied by the data set's `model_factor` when it has one (for example a resolution table that
   the data vector also carries).
5. With r the model vector minus the data and C the covariance of the data set:
   `ln L = -1/2 r^T C^-1 r - 1/2 ln det C - N/2 ln(2 pi)`.

## What is not in it

- **No emulator error.** C is the data set's covariance, as given. The likelihood treats the model as exact, so a
  posterior from it is narrower than one that accounts for the error of the emulator.
- **No bandpower model, no resolution model, no metals, no continuum or DLA-masking systematics** beyond what a data
  set puts into its covariance and its `model_factor`.
- **The two omissions of the mixture** stated on a prediction (`pred.labels["omissions"]`): the power of the subDLA
  cores that the PRIYA filter removes, and the DLA damping-wing core.
- **No prior.** Flat boxes belong to the sampler. Gaussian terms on parameters can be given when the likelihood is
  made; `logp` adds them and `loglike` does not.
- **A point the model refuses raises.** `sampler_logp` returns minus infinity for such a point and counts the
  refusals by kind, so that a chain can say how often and why it was turned back.

## A data set

`P1DData` holds the redshifts, the wavenumbers per redshift (angular, in s/km), the data vector, the full covariance,
an optional factor on the model vector, the truth of a mock and a free-form provenance. `write_p1d_data` writes one
as a strict JSON file with one covariance block per redshift, and `read_p1d_data` reads it only against its sha256.

```python
import numpy as np
from hcd_analysis.likelihood import GaussianP1DLikelihood, P1DData

# emu is a loaded model, params the nine parameters by name (see the README's first example).
theta = dict(params, tau0_amp=1.0, dtau0=0.0)
pred = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0)
k = 0.5 * (pred.k[:-1] + pred.k[1:])            # wavenumbers halfway between the model's modes
n = k.size
blank = P1DData.from_blocks(z=[3.0], k=[k], p1d=[np.ones(n)], cov_blocks=[np.eye(n)])
made_at = GaussianP1DLikelihood(emu, blank).model_vector(theta)
mock = P1DData.from_blocks(z=[3.0], k=[k], p1d=[made_at], cov_blocks=[0.01 * np.eye(n)], truth=theta)
like = GaussianP1DLikelihood(emu, mock)
print("chi2 at the point the mock was made at:", like.chi2(theta))
```

```text
chi2 at the point the mock was made at: 0.0
```

The docstring of `GaussianP1DLikelihood` holds this example, and a test runs it on a synthetic model.

## A stored reference

`P1DReference` holds what a figure needs to show a chain against a reference posterior for a mock, and nothing
else: for each parameter the truth, the posterior mean and the central 68 and 95 per cent intervals; and for each bin
of the data vector some percentiles of the posterior predictive. It holds no chain, no covariance and no likelihood
value. `write_reference` writes one as a strict JSON file and `read_reference` reads it only against its sha256, as
for a data set. `reference.check_bins(data)` refuses a data set whose redshifts and wavenumbers are not exactly the
reference's, and `reference.band(16, 84)` returns two stored percentile rows; nothing is interpolated between stored
levels. How a reference was made is stated in its `provenance`, which the class carries and does not interpret.

## The example: `examples/mock_recovery.py`

```text
python examples/mock_recovery.py --download DOWNLOAD_DIR
python examples/mock_recovery.py --download DOWNLOAD_DIR --demo 200 --out out/
python examples/mock_recovery.py --release RELEASE_DIR SHA256 --data MOCK_FILE SHA256
```

The first two lines read the mock, its model and the stored reference from the example download (next section). The
third runs on a model and a mock of your own, and then has no stored reference.

- **Part 1** prints the chi-square and the log likelihood at the point the mock was made at, and the flags of that
  point. Every line is labelled.
- **Part 2** needs Cobaya, which is not a requirement of the package (`pip install cobaya`). `--demo N` runs one
  short chain of N accepted steps and labels it "A DEMONSTRATION CHAIN, NOT CONVERGED": its numbers are not results.
  `--full` runs one chain with the settings written out in the script until Cobaya's own convergence test stops it.
  Either chain starts at the truth of the mock, and the script says so; it is a single process and refuses to run
  under MPI. A sampler lets INFO records pass, so the predictor's record of the corrections it applies appears in the
  sampler's output: once, because the library writes it once per predictor and at DEBUG afterwards. Without Cobaya
  the part says so and is skipped.
- **Part 3** draws the figures. After a chain: the posterior predictive of the data vector, `posterior_predictive.png`
  (model vectors at points of the chain, each with one replicate of the data drawn from the covariance). With the
  example download the stored reference is drawn underneath in grey (its 2.5 to 97.5 and 16 to 84 per cent bands),
  and a second figure, `posterior_intervals.png`, shows for each of the eleven parameters the reference's mean and
  its 68 and 95 per cent intervals, the truth, and the 16th to 84th percentile of the chain of the run, each on the
  scale of the parameter's flat box. The part also prints the reference's mean and 68 per cent interval of each
  parameter, labelled. Without a chain the figures show the stored reference alone. A `--demo` chain drawn over
  the reference is labelled NOT CONVERGED in both figures: the reference is a converged posterior, a demonstration
  chain is not, and the two are not expected to agree.

What the example fixes is written out at the top of the script:

- **The flat boxes of the eleven parameters:** the parameter limits of Table 2 of Fernandez, Bird and Ho (2024),
  read in the arXiv version. The two helium reionisation redshifts use the package's own sampling box, which is
  narrower than the box the model was trained on. `alphaq` is sampled to 3.0, the edge of the trained box and
  beyond the package's own sampling box (2.5), where the training simulations are sparse. `ns` is sampled to 1.05;
  above 0.995 the training simulations are sparse too, and the predictor flags such a point (`ns_ridge`).
- **Two Gaussian terms,** on `hub` (mean 0.70) and `bhfeedback` (mean 0.05), which this kind of data does not
  constrain. Their standard deviations are 0.015 / sqrt(2) and 0.005 / sqrt(2), narrower by sqrt(2) than the widths that
  Fernandez, Bird and Ho (2024) quote: they are the widths of this project's own reference chains for such a mock.
  The chains are not shipped; the stored reference of the example download is a summary of one of them.
- **The sampler settings of `--full`:** Cobaya's Metropolis sampler, one chain, stopped at R-1 < 0.01 for the means
  and R-1 < 0.05 for the bounds; the first 30 per cent of a chain is dropped before anything is read from it. The
  project's reference chains ran four chains; this script runs one.

A research (fold) model of your own, trained without the simulation your mock comes from, is loaded with
`--research`, its fold MF record with `--mf` and an incidence record built without that simulation with
`--incidence`.

## The example download

**It is not published yet.** Where it will be hosted is decided with the release; until then the example runs only
on a model and a mock of your own. This section says what the download is, so that the script and the files can be
checked against each other when it exists.

The download is one directory. `--download DIR` names it, or the environment variable `HCD_PRIYA_EXAMPLE_DOWNLOAD`
when no model is named on the command line. Its items, each with the sha256 it must have (the values are also
written in the script, as `DOWNLOAD`):

| Item | Place in the directory | What it is | sha256 |
|---|---|---|---|
| `mock` | `mock.p1d.json` | The data set: the P1D of one PRIYA high-resolution simulation, named in the file's provenance, at 9 redshifts and 11 wavenumbers each (99 bins), times a resolution table, with a covariance per redshift derived from the KODIAQ-SQUAD P1D of Karacayli et al. (2022), and the truth. No measured P1D is in it. Its provenance names the inputs | `ae76f028aa482674d59e82e8052164af02bbe20d745b6be2dd1cb810f8fa68f1` |
| `model` | `model/` | A research bundle: a fold model, trained WITHOUT the simulation of the mock. The sha256 is that of its `manifest.json`, which names every other file of the bundle by its own sha256 | `07ae506b4105350081692d460c4b623e9083045a6ebe8130257ba31db5a7d758` |
| `mf_record` | `mf_record/` | The fold's MF record, built without that simulation. The sha256 is that of the record | `667548cfa5a745d5dfb8d7b669d6486860565efa5d78306c21231529ffbeb4c0` |
| `incidence_record` | `incidence_record/` | The fold's incidence record, built without that simulation. The sha256 is that of the record | `33b34ff8df3dcefb25756642dee1cc0c98db944909c79492ec9b24140adbdfe5` |
| `reference` | `reference.json` | The stored reference: a summary of a converged posterior for this mock on this model, and its posterior predictive band | `cdee9267d97cdfe41aa2bc88c0debfacfcb98c6207c20c610ac131352b183599` |

- **The model is not the release model.** It is one fold of the cross-validation, trained without the simulation
  the mock was made from, so that the recovery is a held-out one. It loads only through `load_research_bundle`, with
  a stated purpose. Do not use it for anything but this example.
- **What is checked.** The directory holds a manifest, `example_download.json`, that states the format
  (`hcd_priya.example_download/1`) and the five sha256. The script refuses a directory whose manifest states
  anything else: it was made for another version of the script. The manifest is a first check only. What protects
  the read is that each loader refuses an item whose sha256 is not the one written in the script, and that the
  reference must be on the bins of the mock, state the mock's truth and hold the percentile levels the figures
  draw. Every such refusal ends the script with one line that starts with "refusing:".
- **What the stored reference holds, and nothing more:** the truth of the mock, the posterior mean and the central
  68 and 95 per cent intervals of the eleven parameters, and the percentiles 2.5, 16, 50, 84 and 97.5 of the
  posterior predictive on the 99 bins. It comes from the project's own reference chains for this mock: Cobaya's MCMC
  sampler, four chains, stopped at R-1 below 0.01 for the means and below 0.05 for the 95 per cent bounds, the first
  30 per cent of each chain dropped; the flat boxes and the two Gaussian terms of the example; the model and the mock
  of the download. The predictive percentiles are over 1,000 posterior draws of a replicate of the data vector (the
  model vector at the draw plus noise from the data covariance). The chains were sampled with the project's private
  reference likelihood, not with this module; the two were compared on this model and this mock and agree (to
  rounding at the truth, and at the printed precision of the reference chain file at 32 points of the prior box).
- **What the reference is not.** No emulator error is in the covariance, so its intervals are narrower than ones
  that account for the error of the emulator. It is one mock on one model: it says that the pieces recover a known
  point of a simulation, and nothing about any measurement.

## A loader for the XQ-100 P1D, and nothing more

`load_xq100(data_file, covariance_file, cross_redshift=...)` reads the XQ-100 Lyman-alpha P1D and its covariance
from two files that you supply into a `P1DData`: five redshifts, 3.4 to 4.2, with 13 wavenumbers each.

- **Nothing is shipped.** The two file names and the sha256 each must have are in `likelihood.XQ100_FILES`; a file
  with another digest is refused. In November 2026 the data file is public in `sbird/lya_emulator` and in
  `jibanCat/InferenceLyaData` under `lyaemu/data/xq100/`, and the covariance file in the second only. Neither
  repository states a licence.
- **Nothing is fitted, and no result of this package rests on these data.** A likelihood made from the returned
  object has none of the terms a real analysis needs: no emulator error, no resolution or metal model, no inflation
  of the low wavenumbers.
- **`cross_redshift` has no default.** `True` keeps the full 65 x 65 matrix of the file; `False` keeps the five
  blocks on the diagonal and sets the covariance between redshifts to zero. The choice changes a likelihood, so it
  is the user's to state.
- **Three things are not verified here.** The code the files are distributed with attributes them to the XQ-100
  analysis of Wilson, Irsic and McQuinn (2021); this project has not verified that attribution, or the definition
  of the column it reads, against the paper. And the covariance file holds indices and values only: that its order
  is redshift 3.4 to 4.2 ascending, then wavenumber ascending, is taken from the same code, which marks it as
  unchecked. A wrong order would give a wrong likelihood without any error.

## References

Karacayli, N. G., Padmanabhan, N., Font-Ribera, A., et al. 2022, "Optimal 1D Ly-alpha Forest Power Spectrum
Estimation - II. KODIAQ, SQUAD & XQ-100", [arXiv:2108.10870](https://arxiv.org/abs/2108.10870), related DOI
10.1093/mnras/stab3201 (title and authors read from the arXiv page; the journal fields were not verified).

Fernandez, M. A., Bird, S., Ho, M.-F. 2024, "Cosmological constraints from the eBOSS Lyman-alpha forest using the
PRIYA simulations", JCAP 2024 (07) 029, [arXiv:2309.03943](https://arxiv.org/abs/2309.03943).

Wilson, B., Irsic, V., McQuinn, M. 2021, "A measurement of the Lyman-beta forest power spectrum and its cross with
the Lyman-alpha forest in X-Shooter XQ-100", [arXiv:2106.04837](https://arxiv.org/abs/2106.04837) (title and authors
read from the arXiv page; the journal fields were not verified).
