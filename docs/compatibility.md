# Supported and tested environments

**Supported** means the CI tests pass there and a problem found there is treated as a bug of this project. At this
stage only one environment is supported; wider support will be considered before a production release.

| Environment | Status |
|---|---|
| Linux x86-64, CPython 3.11, the versions pinned in `tools/ci/constraints-devenv.txt` | **Supported.** The CI job "Python 3.11, pinned (supported)" installs these versions; the packages the file does not pin (click, joblib, pandas, pyyaml and some of their dependencies) take the newest allowed version. |
| Linux x86-64, CPython 3.11, the newest versions the dependency bounds allow | Tested in CI, **not supported**. |
| Linux x86-64, CPython 3.12, the newest versions the dependency bounds allow | Tested in CI, **not supported**. |
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

## Installing the supported environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -c tools/ci/constraints-devenv.txt ".[test]"
```

`fake_spectra`, needed only to build catalogs and P1D from raw simulation spectra, is installed separately
(`docs/fake_spectra_integration.md`) and is not covered by this table.
