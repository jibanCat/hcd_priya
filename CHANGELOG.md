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

### Removed
- Historical analysis and campaign scripts, batch scripts, write-ups, figures and old manifests that neither the
  package nor its tests use were removed from the tree. They remain in the Git history; documents that still name
  them refer to that history.

## 0.1.0

The research code before it was packaged for release.
