# hcd_priya: a Lyman-α forest P1D emulator with high-column-density absorber classes

![Python](https://img.shields.io/badge/python-3.11-blue)
![JAX](https://img.shields.io/badge/JAX%20%2F%20Equinox-float64-orange)
[![tests](https://github.com/jibanCat/hcd_priya/actions/workflows/tests.yml/badge.svg)](https://github.com/jibanCat/hcd_priya/actions/workflows/tests.yml)

This repository holds the code of an emulator of the one-dimensional flux power spectrum
(P1D) of the Lyman-α (Lyα) forest that models the high-column-density (HCD)
absorbers contaminating it, together with the research pipeline it was built
for: measuring two cosmological parameters, the forest power amplitude A_p and
the tilt n_s, from the P1D. The supported interface of this release is the
predictor of the per-class P1D; the measurement itself is research code that does
not run at this commit (see
[Scope and limitations of this release](#scope-and-limitations-of-this-release)).
It is built on the PRIYA suite of cosmological hydrodynamic simulations.

---

## Install

<!-- docs:install:start -->
`hcd-priya` installs the Python package `hcd_analysis` and the command `hcd`. It requires `hcd-priya-models`, a
data-only package that holds the trained models. It needs Python 3.11 or later on Linux (x86-64).

**What works today: an install from a clone.** Neither package is on a package index yet. Install from a clone of
this repository, and give the wheel of the model package as a file (given as a file, it is never looked up on an
index):

```bash
git clone https://github.com/jibanCat/hcd_priya && cd hcd_priya
python3.11 -m venv .venv && source .venv/bin/activate
pip install . <path to the hcd_priya_models wheel>
python -c "import hcd_analysis; print(hcd_analysis.__version__)"
hcd --help
```

- **Which model wheel.** No trained model is published yet. The only model wheel that someone outside the project
  can build today is a stand-in with a synthetic bundle: untrained networks, which show that the installation
  works and whose numbers are not science. The model wheel is built in this repository
  ([`model_package/`](model_package/README.md)); `tools/ci/install_with_synthetic_models.sh` builds the stand-in and
  installs it ([`CONTRIBUTING.md`](CONTRIBUTING.md) has the commands). With the stand-in, the first example of the
  README runs only with `HCD_PRIYA_ALLOW_SYNTHETIC=1` set, and with a warning.
- **The supported install** pins every dependency to the versions the test suite is developed and run with
  ([`tools/ci/constraints-devenv.txt`](tools/ci/constraints-devenv.txt)), on CPython 3.11:

  ```bash
  pip install -c tools/ci/constraints-devenv.txt ".[test]" <path to the hcd_priya_models wheel>
  ```

**What will work after publication (not available yet).** Once both packages are published, the whole install will
be one line. It does not work today:

```bash
pip install hcd-priya        # NOT AVAILABLE YET: neither package is on a package index
```

It will install the code and the trained LF+MF model together, and the first example of the README will then run as
written, with no switch and no warning.

**Notes.**

- Newer dependency versions and Python 3.12 are tested in continuous integration but not supported yet; macOS and
  Windows are not tested. [`docs/compatibility.md`](docs/compatibility.md) lists what is tested and what is not.
- **Extras:** `test` (pytest), `tutorials` (jupyter, nbformat) and `dev`. `pyproject.toml` declares the
  dependencies; `requirements.txt` only points to the install above.
- **Working in a clone.** To run the scripts or the pipeline from the clone, install it editable (add `-e` before
  `".[test]"` in the supported install). The pipeline records the clone's commit, and only
  an editable install makes that commit the code that runs (with your changes committed); a plain install runs the
  copy made when it was installed. The scripts under `scripts/` are not part of the installed package; run them from
  the repository root.
- **JAX on the CPU, in double precision.** Importing `hcd_analysis.emulator` enables float64, which several
  structural identities of the model need: they hold only to bit-level accuracy and break under JAX's default
  float32. Run scripts with `JAX_PLATFORMS=cpu python <your_script.py>`.
- **The simulation side** (building absorber catalogs and P1D from raw spectra) also needs `fake_spectra`, installed
  separately ([`docs/fake_spectra_integration.md`](docs/fake_spectra_integration.md)); the predictor does not. On
  the Great Lakes cluster, `bash scripts/install_greatlakes.sh` creates a conda environment with both.
<!-- docs:install:end -->

---

## Quickstart

Load the LF+MF model and predict the P1D of each class of sightlines at one redshift. The nine parameters of the
model are written out, so that you see which point the prediction is for:

```python
import hcd_analysis.predictor as hp

emu = hp.load_published()        # the LF+MF model, from the installed model package

# The nine PRIYA parameters, by name. These values are the midpoint of the range each parameter
# was trained on: a point to try the model at. It is not a fiducial cosmology, not a best fit
# and not a simulation. Put your own values here.
params = {
    "ns": 0.925,           # slope of the primordial power spectrum at k = 0.78 / Mpc
    "Ap": 1.9e-9,          # its amplitude at the same wavenumber
    "herei": 4.0,          # redshift at which helium reionization starts
    "heref": 2.7,          # redshift at which helium reionization ends
    "alphaq": 2.15,        # quasar spectral index, which sets the heating of helium reionization
    "hub": 0.70,           # dimensionless Hubble parameter h
    "omegamh2": 0.143,     # Omega_m h^2
    "hireionz": 7.25,      # redshift of hydrogen reionization
    "bhfeedback": 0.05,    # black-hole feedback strength
}

# z is a PRIYA snapshot redshift. The mean flux is PRIYA's model at its centre:
# tau0_amp = 1 and dtau0 = 0 give the fit of Kim et al. (2007).
pred = emu.predict(params, z=3.0, tau0_amp=1.0, dtau0=0.0)

print("model:", pred.labels["product"])
print("classes, in the order of P_class:", hp.CLASSES)
print("shape of pred.P_class, the P1D in km/s of each class:", pred.P_class.shape)
print("shape of pred.k, the wavenumbers in s/km:", pred.k.shape)
print("flags that are set (read them before you use the numbers):", sorted(n for n, v in pred.flags.items() if v is True))
```

With the LF+MF model this prints (the model has 172 wavenumbers; no spectrum value is shown here):

```text
model: LF+MF
classes, in the order of P_class: ('clean', 'LLS', 'subDLA', 'DLA')
shape of pred.P_class, the P1D in km/s of each class: (4, 172)
shape of pred.k, the wavenumbers in s/km: (172,)
flags that are set (read them before you use the numbers): ['k_downweighted', 'mf_applied']
```

The two flags say that the multi-fidelity correction was applied, and that some of the model's own wavenumbers lie
below 1e-3 s/km, where training gave less weight.

- **What "LF+MF" is.** The emulator trained on the low-fidelity (low-resolution) PRIYA simulations, with the
  multi-fidelity correction that calibrates it to the high-resolution simulations. It is the model this package
  publishes, and `hp.load_published()` loads it without a name.
- **Where the model comes from.** The trained model is in a companion package, `hcd-priya-models`, which `hcd-priya`
  requires. Nothing is downloaded when you load it. **Neither package is published yet**: [Install](#install) says
  what works today, and [Models and data](#models-and-data) what `load_published` checks. With a release's files
  in hand, the same model loads from its directory and the sha256 published with it:

  ```python
  emu = hp.load_release("path/to/release", "<published release sha256>")
  ```

What goes in and what comes out:

- **Parameters.** `params` is a dict of the nine PRIYA parameters by name (`hp.PARAM_NAMES`), as above. A value
  outside the range the model was trained on (`hp.PARAM_LIMITS`) is predicted and flagged, not refused.
- **Redshift.** `z` is one of the PRIYA snapshot redshifts 2.0, 2.2, ..., 5.4; `emu.predict_many` takes several.
- **Mean flux.** Give either `tau0_amp` and `dtau0` (PRIYA's model), or `tau_eff` from any other mean-flux model
  (`help(hp.tau_eff_priya)` shows how, with a published fit as the example). A mean flux outside the range the model
  was trained on is predicted with a warning and a flag.
- **Result.** `pred.k` (s/km), `pred.P_class` (clean, LLS, subDLA, DLA), `pred.w_c` and `pred.dndx` (the model's
  sightline fractions and incidence of the three HCD classes). `hp.add_hcd(pred, pred.w_c)` mixes the classes into one
  P1D. **Read `pred.flags` before using the numbers**: a parameter outside the training box, or an LF+MF prediction
  above z 4.6, is returned and flagged, not refused.

Without any model file, the same calls run on a synthetic bundle (untrained networks, so it shows the interface and
behaviour, not science): `hcd_analysis.emulator.release_testing.write_synthetic_bundle` writes one that
`hp.load_bundle` loads, and the tutorial
[`docs/tutorials/production_and_research_bundles.md`](docs/tutorials/production_and_research_bundles.md) walks
through production and research bundles on synthetic data. Every function named above has a docstring with an
example that a test runs (for instance `help(hp.Prediction)`).

---

## What this is

<!-- docs:about:start -->
The Lyα forest is the series of absorption lines seen blueward of a distant
quasar's Lyα emission, each produced by intergalactic hydrogen along the line of
sight. The statistics of that absorption, and in particular its one-dimensional
power spectrum P1D(k), trace the small-scale clustering of matter, which in turn
constrains the primordial power spectrum (A_p, n_s) together with the nature of
dark matter and neutrinos.

A fraction of sightlines, however, intersect dense gas systems. These are the
Lyman-limit systems (LLS), the sub-damped and damped Lyα absorbers (subDLA and
DLA), known collectively as HCDs. They imprint broad damped wings on the
spectrum and bias the measured P1D if they are not modelled. We split the three
classes by neutral-hydrogen column density N_HI (`hcd_analysis/catalog.py`):
LLS at 10^17.2–10^19, subDLA at 10^19–10^20.3, and DLA at N_HI ≥ 10^20.3 cm⁻².
The pipeline measures P1D from the PRIYA simulations and trains a fast
differentiable emulator that carries an explicit free-amplitude model of the HCD
contamination. The research code then feeds that emulator into a gradient-based
likelihood, with the aim of inferring (A_p, n_s) while marginalising over the HCD
incidence; that last step is not part of this release. Treating
the HCDs correctly is what allows a forest P1D analysis to remain unbiased at the
sub-percent precision that DESI now delivers.

PRIYA: Bird, Fernandez, Ho et al. 2023, JCAP 10 037
([arXiv:2306.05471](https://arxiv.org/abs/2306.05471)); extended box from
Fernandez, Bird & Ho 2024, JCAP 07 029
([arXiv:2309.03943](https://arxiv.org/abs/2309.03943)).

hcd_priya is the continuation of the analysis of Ho et al. 2026,
"Small-scale Lyman alpha forest cosmology with PRIYA: constraints from XQ100 and KODIAQ-SQUAD one-dimensional flux
power spectra", JCAP 2026 (07) 094
([doi:10.1088/1475-7516/2026/07/094](https://doi.org/10.1088/1475-7516/2026/07/094),
[arXiv:2509.18271](https://arxiv.org/abs/2509.18271)). How to cite the software: [`CITATION.cff`](CITATION.cff).
<!-- docs:about:end -->

---

## Scope and limitations of this release

The supported public interface of this release is the predictor (`hcd_analysis.predictor`): the per-class P1D
(clean, LLS, subDLA, DLA) from an LF+MF model, and a linear mix of the HCD classes (`add_hcd`). The rest of the
repository (the catalogue pipeline, training, the likelihood and the fit scripts) is research code outside that scope.

Not in this release:

- **No cosmological inference.** No fit, sampler driver or supported likelihood runs from this repository at this
  commit (Stages 3 and 4 of [`docs/pipeline.md`](docs/pipeline.md); [`tests/DEFERRED.md`](tests/DEFERRED.md)).
- **No HCD fitting, and no model of how the HCD content changes with redshift.** `add_hcd` evaluates any physical
  amplitudes, one per class per redshift, and fits nothing; the incidence law and its priors are research code.
- **No prediction between snapshot redshifts.** Predictions exist at z = 2.0, 2.2, ..., 5.4 only; an LF+MF prediction
  is refused below the MF record's lower edge (z 2.2 by default) and flagged above z 4.6.
- **No bandpower model.** A prediction is a per-class spectrum at points in k: no metal lines, no
  spectrograph-resolution correction, no bin window, and no helper from a measured data leg to a model vector.
- **No calibrated emulator error.** `member_std` is the spread over ensemble members that differ only in their random
  seed; it is not an error bar and not a covariance.
- **No trained model, catalogue or derived product of this project is public yet** (see
  [Models and data](#models-and-data)).

---

## Models and data

<!-- docs:data:start -->
**Data availability.** The data products of this project are HCD absorber catalogues and the products derived from
them; it does not provide raw simulation data. Researchers who need the raw PRIYA simulation outputs can request them
from the authors of Bird et al. 2023, JCAP 2023 (10) 037
([doi:10.1088/1475-7516/2023/10/037](https://doi.org/10.1088/1475-7516/2023/10/037),
[arXiv:2306.05471](https://arxiv.org/abs/2306.05471)). The catalogues and derived products are not in this code
repository, and their release is planned but not yet available.
<!-- docs:data:end -->

**Trained models.** They come in the companion package `hcd-priya-models` (data only, MIT licence), which
`hcd-priya` requires. `hp.load_published()` finds the model in the installed package and loads it only against
the sha256 that `hcd-priya` itself lists for that model version (`hcd_analysis/published_models.py`): for a
published model the digest comes from the code package, never from the model package, so changed model files are
refused, and so is a model version this version of the code does not list. No model is published yet, and that list is empty.
`hp.load_release(release_dir, sha256)` loads a release whose files you hold yourself. To check a model from a fresh
install, run `python -I -m hcd_analysis.check_release --published`, or with `RELEASE_DIR SHA256` for a release
directory (`-I` keeps a clone in the current directory out of the import path, so the installation is what is
tested).

The details are in [`docs/models_and_data.md`](docs/models_and_data.md): what is and is not distributed, the release
format (HDF5 + JSON, no pickle), the LF+MF correction, and the environment variables of the observed P1D files that
the research code reads.

---

## Where to go next

- To report an issue, ask for help or contribute: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- The API reference of the predictor and the loaders is on the documentation site, built from
  [`docs/api.md`](docs/api.md); a step-by-step guide to production and research bundles with executable examples
  is [`docs/tutorials/production_and_research_bundles.md`](docs/tutorials/production_and_research_bundles.md).
- The research pipeline (catalogue construction, and the four stages from the simulation P1D to inference, a
  historical record) is described in [`docs/pipeline.md`](docs/pipeline.md).
- The emulator deep-dive covers the architecture, the τ₀ cache, the training
  recipe, the forward model, the input contract and the conventions:
  [`hcd_analysis/emulator/README.md`](hcd_analysis/emulator/README.md).
- The tutorials are five short notebooks that introduce the per-(sim, snap) HCD
  data products from the ground up, with a recommended reading order:
  [`notebooks/tutorials/README.md`](notebooks/tutorials/README.md).
- The science walkthrough and the record of bugs found on the catalog/P1D side
  are in [`docs/analysis.md`](docs/analysis.md),
  [`docs/bugs_found.md`](docs/bugs_found.md),
  [`docs/data_layout.md`](docs/data_layout.md) and
  [`docs/p1d_definition.md`](docs/p1d_definition.md).
- The validation results (held-out LOSO error, coverage and the Fisher checks)
  are not public yet; the closure gates belong to the research path and are not
  part of this release.

---

## Tests / status

Install the test extra (above), then run a test file with pytest, one file per process:

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_cli_config.py
```

Many tests need real assets that are not in this repository (the training cache, trained emulator members,
observational data, raw simulation outputs). [`tests/ci_tiers.toml`](tests/ci_tiers.toml) assigns every test
file to a tier and says why a file or test is not run in continuous integration. The workflow
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) builds the wheel and the source archive, checks
their contents against the commit, installs the wheel into a fresh environment and runs the "wheel" and "source"
tiers with [`tools/ci/run_test_tier.py`](tools/ci/run_test_tier.py), which reports every skipped, deferred or
deselected test as not run, never as passed. The same runner works from a clone:

```bash
JAX_PLATFORMS=cpu python tools/ci/run_test_tier.py --tier source --out <new_results_dir>
python tools/ci/run_test_tier.py --list-not-run
```

Status: research code under active development. The validation results are not public yet.

---

## Repository map

```
hcd_analysis/            catalog/P1D/CDDF pipeline (catalog.py, p1d.py, cddf.py, hcd_template.py, ...)
hcd_analysis/emulator/   the JAX/Equinox emulator + likelihood + inference (see its own README)
hcd_analysis/cli/run.py  the `hcd` command-line entry point
config/default.yaml      default pipeline config
scripts/                 install, SLURM batch, cache build, LOSO sweep, plotting
notebooks/tutorials/     five starter notebooks for new students
tests/                   science-claim regression tests
docs/                    analysis.md, bugs_found.md, data_layout.md, handover, ...
figures/analysis/        all analysis + validation figures
checkpoints/             trained emulator bundles (final_fold0.*) + error_vector.npz; gitignored, not in this repo (see docs/models_and_data.md)
hcd_analysis/_emulator_data/  the 1.1 GB τ₀ cache; gitignored, not in this repo (see docs/models_and_data.md)
```
