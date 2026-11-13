# Changelog

All notable changes to hcd_priya. Versions follow `hcd_analysis.__version__`.

## 0.2.0 (unreleased)

### Fixed
- The absorber catalogue no longer lists a system that straddles the box edge twice
  ([`docs/bugs_found.md`](docs/bugs_found.md), bug 8). An independent second finder, an overlap check and a
  three-way count reconciliation now guard every catalogue, and a cached catalogue made before the fix is refused on
  reuse.
- An LF+MF prediction above z 4.6, the upper edge of the frozen grouped-validation domain (z 2.2-4.6; the held-out validation is reported separately), is now
  flagged `mf_z_above_validated` (values unchanged; the record format is unchanged).

### Added
- **Release format `hcd_priya.h5/2`** for emulator bundles: HDF5 + JSON, pickle-free, with a lineage block that ties
  a bundle to its training cache, catalogues and code. The release loader refuses a bundle whose lineage it cannot
  establish ([`docs/release_format_h5.md`](docs/release_format_h5.md)).
- **A public predictor** (`hcd_analysis.predictor`): per-class P1D, HCD templates and the class weights w_c from a
  clean-lineage bundle, with free HCD amplitudes and no informative HCD priors.
- **Production and research bundles kept apart.** A bundle trained without some simulations (a validation fold) can
  only be loaded through the explicit research path.
- **The trained models come in a companion package, `hcd-priya-models`** (`model_package/`; data only, MIT
  licence), which `hcd-priya` now requires. **`load_published`** (`hcd_analysis.predictor`) loads a model by name
  from the installed model package, only against the sha256 listed in `hcd_analysis/published_models.py`.
  With no argument it loads the LF+MF model; a name is compared without case and punctuation. The examples write
  the nine parameters out; `example_params` gives the midpoint of the training range of each parameter and prints
  that statement and the values when it is called. No model is published yet, so the list is empty; tests
  and CI use a model package built with a synthetic bundle.
- **Three figures with their scripts and data** ([`docs/model_figures.md`](docs/model_figures.md)): the model's
  incidence dN/dX and its column-density distribution against the DLA measurements of Ho, Bird and Garnett (2021),
  and the class power ratio
  of the training simulations against Eq. 6 of Rogers et al. (2018). `scripts/figures/` draws them; the data files
  are in `docs/figures/data/`, each with its origin and sha256. The class-ratio figure regenerates from the
  repository alone. The dN/dX and CDDF figures need a model with an incidence record (none is published yet);
  their measured points and the 60 PRIYA design points of their bands are committed figure data.
- **A warning for a mean flux outside the trained range.** A prediction whose `tau_eff` lies outside the range of
  mean flux the model was trained on, or outside the tables of the multi-fidelity correction, emits a
  `MeanFluxRangeWarning` that gives the range. The prediction is still
  returned, with the flag `mean_flux_off_ladder` as before.
- **The multi-fidelity correction of the incidence.** A release can carry an incidence MF record
  (`hcd_priya.incidence_mf_record/1`, an optional named part of the release manifest; the manifest keeps its one
  format, `hcd_priya.release/1`, and a manifest without the part is unchanged): the
  fixed HR/LF ratio of dN/dX per HCD class and redshift, built by the functions that `multifidelity.py` has held
  since June 2026 and by the new `scripts/build_incidence_mf.py`. With it, an LF+MF prediction's `dndx` is
  multiplied by that ratio and `w_c` follows through the shipped map (flag `incidence_mf_applied`). A release of
  format `hcd_priya.release/1` loads and predicts as before. No such record is released yet.
- **The column-density distribution as an output.** With an incidence record a prediction carries `cddf` and
  `log_nhi`: the CDDF at the record's bin centres, the linear mean over the ensemble members, and under LF+MF that
  mean times the record's HR/LF ratio per bin. A bin is NaN, never a number, where no HR/LF simulation pair has
  absorbers at any tabulated redshift, and under LF+MF also where none has them at the redshift used.
  Without an incidence record both are `None`.
- **A documentation page on the class spectra against the per-class model of Rogers et al. (2018)**
  ([`docs/rogers_comparison.md`](docs/rogers_comparison.md)): what was compared, the numbers, and what the
  comparison is not.
- **`python -m hcd_analysis.check_release`**: loads a model from a fresh install, predicts, and compares with
  reference predictions.
- **Examples in the docstrings of the user entry points**, run by a test; type annotations on the loaders,
  `predict`, `predict_many` and `add_hcd`.
- **Runtime data inputs** (`hcd_analysis.runtime_data`): the observed P1D legs and the multi-fidelity
  resolution-correction table come from an explicit path or an environment variable, never a built-in path, and are
  verified by sha256 before use; the simulation roots come from the config file or `HCD_LF_RAW_DIR` /
  `HCD_HR_RAW_DIR`, and an empty or missing root is refused. An empty path or variable is refused; an explicit DESI
  path is checked as the sample `use_snr3` selects, and the other sample's file is refused. Loaded legs carry a record
  of each verified file (`data_products`), and the likelihood's entry points print each verified load
  (`runtime_data.show_loads`).
- **A run-settings record** in each output directory: a rerun with different settings is refused instead of mixing
  results.
- **A model registry** (`hcd_analysis.registry`), provisional: download, sha256 check, cache and offline use for
  released models. No storage format is mapped and no model is listed yet.
- **Packaging:** the distribution `hcd-priya`, the `hcd` command under `hcd_analysis.cli`, and declared dependencies,
  with `numpy>=2.0`.
- **Continuous integration** ([`docs/compatibility.md`](docs/compatibility.md)):
  - the supported, pinned environment (Linux, Python 3.11), plus newer-dependency jobs that are tested but not
    supported;
  - a check that the built wheel and source archive contain exactly the committed package;
  - a scan of what the built wheel and source archive contain for private paths, internal pointers, credentials
    and confidentiality markers: CI fails on a new finding, and a release needs none
    ([`docs/release_checklist.md`](docs/release_checklist.md));
  - test tiers in which every skip must be declared;
  - an older-libhdf5 read test of the release format.
- **Project files:** [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) (Contributor
  Covenant 2.1) and [`CITATION.cff`](CITATION.cff).
- **A documentation site** for Read the Docs (Sphinx with MyST): install, data availability, supported environments,
  the release format, the production and research bundles tutorial and an API reference of the public modules. CI
  builds it with every warning treated as an error.

### Changed
- The `hcd` command needs an explicit `--config` file and refuses to run without one. Keys the file leaves out
  still take the built-in values of `hcd_analysis.config`, and keys it does not know are ignored.
- The install name is `hcd-priya`; the import name stays `hcd_analysis`.
- The README is short: install, load a model, predict. The release format, the runtime data inputs and the
  description of the research pipeline moved, word for word, to `docs/models_and_data.md` and `docs/pipeline.md`.
- The Python code is formatted with black (line length 100), with no change in behaviour. Catalogues built from
  now on record different finder source digests than earlier catalogues built from the same code, and a cache
  carry needs a newly reviewed row-path diff ([`CONTRIBUTING.md`](CONTRIBUTING.md), "What a reformat changes").

### Removed
- Historical analysis and campaign scripts, batch scripts, write-ups, figures and old manifests that neither the
  package nor its tests use were removed from the tree. They remain in the Git history; documents that still name
  them refer to that history.

## 0.1.0

The research code before it was packaged for release.
