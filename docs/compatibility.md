# Supported and tested environments

**Supported** means the CI tests pass there and a problem found there is treated as a bug of this project. At this
stage only one environment is supported; wider support will be considered before a production release.

| Environment | Status |
|---|---|
| Linux x86-64, CPython 3.11, the versions pinned in `tools/ci/constraints-devenv.txt` | **Supported.** The CI job "Python 3.11, pinned (supported)" installs these versions; the packages the file does not pin (click, joblib, pandas, pyyaml and some of their dependencies) take the newest allowed version. |
| Linux x86-64, CPython 3.11, the newest versions the dependency bounds allow | Tested in the nightly CI, **not supported**. |
| Linux x86-64, CPython 3.12, the newest versions the dependency bounds allow | Tested in the nightly CI, **not supported**. |
| CPython 3.13 or later | Not tested. `requires-python` does not exclude it. |
| macOS, Windows, other CPU architectures, GPUs | Not tested. |
| The HDF5 release format read with an older libhdf5 | Tested in CI: a bundle written with the pinned stack is read bit for bit with h5py built against Ubuntu's system libhdf5 1.10, and the loader's tests pass there (job "HDF5 release format with an older libhdf5"). |
| The lower bounds in `pyproject.toml` | Tested only where a bound equals the pinned version (jax, equinox, optax, numpyro, scipy, h5py, matplotlib, pytest, at the pinned patch release). numpy's bound is 2.0 because jax 0.10.1 requires numpy 2.0 or later and the package uses nothing newer; no CI job installs numpy 2.0 itself. The bounds of pandas, pyyaml, click and joblib are not tested. |

A job that is tested but not supported may fail. Its failures are shown, not hidden: no CI job is allowed to fail
silently. The current result of each job is on the repository's Actions page; a run on a developer's machine is not
a substitute for it.

## What the CI jobs check

Each job builds the wheel and the source archive, checks that they contain exactly the committed package, installs
the wheel into a fresh virtual environment and runs two tiers of tests (`tests/ci_tiers.toml`):

- **wheel:** tests of the installed package, run from a directory that holds only `tests/`;
- **source:** tests that also read repository files, run from the checkout (they import the package from the
  checkout, not from the wheel).

Tests that need real assets (the training cache, trained emulator members, observational data, raw simulation
outputs) are not run in CI. They are listed with their reasons in `tests/ci_tiers.toml` and reported as not run.

### Three levels: pull request, nightly, release

Every test file also has a level in `tests/ci_tiers.toml`, which says when it runs:

- **pr:** every pull request and every push to `main` (`.github/workflows/tests.yml`), on the supported pinned stack
  only. It holds the tests of what a user runs (loading, prediction, the likelihood, the release formats and their
  writers) and the tests that check a scientific result against an independent computation in five areas: the
  likelihood, the covariance, normalisation, the predictor's reconstruction and the incidence model.
  `tests/test_ci_tiers.py` lists those tests and fails if one of them leaves this level.
- **nightly:** the pr level and the remaining CI tests, once a day and by hand (`.github/workflows/nightly.yml`), on
  the pinned stack and on the two newer-dependency stacks above. A failure fails the run, visibly.
- **release:** the tests that need the real assets. They run only where the assets are, before a release (`python
  tools/ci/run_test_tier.py --tier assets`), and the record of that run is attached to the release.

Within a file, single tests can differ from the file's level; `tests/ci_tiers.toml` names them. Nightly failures
block a release: the release validation workflow (`.github/workflows/release-validation.yml`, started by hand) and
the publishing workflow refuse to go on unless the latest nightly run on the commit to be released passed.

Floating-point results are compared with stored reference arrays at stated tolerances, so a verdict does not
depend on the last bits of the runner CPU's arithmetic: the tolerances are larger than the differences measured
between the CPUs tried. The bit-level digests of three test files are never checked in CI: they run only in an
opt-in replay with the pinned stack on the kind of CPU they were recorded on, x86-64 with AVX-512
(`HCD_EXACT_REPLAY=1`; see `docs/release_checklist.md`). On another kind of CPU a digest failure is inconclusive.

## Installing the supported environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -c tools/ci/constraints-devenv.txt ".[test]" <path to the hcd_priya_models wheel>
```

`hcd-priya` requires the model package `hcd-priya-models`, which is not on the package index yet; the README's
install section says where its wheel comes from.

To run the scripts or the pipeline from a clone, install it editable (`-e`), as `CONTRIBUTING.md` does: the pipeline
records the clone's commit, and only an editable install makes that the code that runs (with the changes
committed).

`fake_spectra`, needed only to build catalogs and P1D from raw simulation spectra, is installed separately
(`docs/fake_spectra_integration.md`) and is not covered by this table.
