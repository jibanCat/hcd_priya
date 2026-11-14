# The Gaussian likelihood and the mock-recovery example

`hcd_analysis.likelihood` turns a loaded model and a P1D data set into a log likelihood. This page says what it
computes, what it leaves out, and how the example that comes with it is run. The reference of every function is on
the [API page](api.md).

**Nothing is fitted to a measurement by anything on this page.** The example recovers the parameters of a mock, a
data vector made from a simulation at a known point. No mock and no model is shipped yet; the example takes both
as files you name.

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

## The example: `examples/mock_recovery.py`

```text
python examples/mock_recovery.py --release RELEASE_DIR SHA256 --data MOCK_FILE SHA256
python examples/mock_recovery.py --release RELEASE_DIR SHA256 --data MOCK_FILE SHA256 --demo 200 --out out/
```

- **Part 1** prints the chi-square and the log likelihood at the point the mock was made at, and the flags of that
  point. Every line is labelled.
- **Part 2** needs Cobaya, which is not a requirement of the package (`pip install cobaya`). `--demo N` runs one
  short chain of N accepted steps and labels it "A DEMONSTRATION CHAIN, NOT CONVERGED": its numbers are not results.
  `--full` runs the settings written out in the script until Cobaya's own convergence test stops it. Without Cobaya
  the part says so and is skipped.
- **Part 3**, after a chain, draws the posterior predictive of the data vector: model vectors at points of the
  chain, each with one replicate of the data drawn from the covariance.

What the example fixes is written out at the top of the script:

- **The flat boxes of the eleven parameters:** the parameter limits of Table 2 of Fernandez, Bird and Ho (2024),
  read in the arXiv version. The two helium reionisation redshifts use the package's own sampling box, which is
  narrower than the box the model was trained on. `alphaq` is sampled to 3.0, the edge of the trained box and
  beyond the package's own sampling box (2.5), where the training simulations are sparse.
- **Two Gaussian terms,** on `hub` (mean 0.70) and `bhfeedback` (mean 0.05), which this kind of data does not
  constrain. Their standard deviations are 0.015 / sqrt(2) and 0.005 / sqrt(2), the widths of the reference chains the
  example is compared with; they are narrower by sqrt(2) than the widths that Fernandez, Bird and Ho (2024) quote.
- **The sampler settings of `--full`:** Cobaya's Metropolis sampler, four chains, stopped at R-1 < 0.01 for the means
  and R-1 < 0.05 for the bounds; the first 30 per cent of a chain is dropped before anything is read from it.

A research (fold) model, trained without the simulation the mock comes from, is loaded with `--research`, its fold
MF record with `--mf` and an incidence record built without that simulation with `--incidence`.

## References

Fernandez, M. A., Bird, S., Ho, M.-F. 2024, "Cosmological constraints from the eBOSS Lyman-alpha forest using the
PRIYA simulations", JCAP 2024 (07) 029, [arXiv:2309.03943](https://arxiv.org/abs/2309.03943).
