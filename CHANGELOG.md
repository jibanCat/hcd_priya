# Changelog

All notable changes to hcd_priya. Versions follow `hcd_analysis.__version__`.

## 0.2.0 (unreleased)

### Fixed
- The absorber catalogue no longer lists a system that straddles the box edge twice
  ([`docs/bugs_found.md`](docs/bugs_found.md), bug 8). An independent second finder, an overlap check and a
  three-way count reconciliation now guard every catalogue, and a cached catalogue made before the fix is refused on
  reuse.

### Added
- **Release format `hcd_priya.h5/2`** for emulator bundles: HDF5 + JSON, pickle-free, with a lineage block that ties
  a bundle to its training cache, catalogues and code. The release loader refuses a bundle whose lineage it cannot
  establish ([`docs/release_format_h5.md`](docs/release_format_h5.md)).
- **A public predictor** (`hcd_analysis.predictor`): per-class P1D, HCD templates and the class weights w_c from a
  clean-lineage bundle, with free HCD amplitudes and no informative HCD priors.
- **Production and research bundles kept apart.** A bundle trained without some simulations (a validation fold) can
  only be loaded through the explicit research path.
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
  - test tiers in which every skip must be declared;
  - an older-libhdf5 read test of the release format.
- **Project files:** [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) (Contributor
  Covenant 2.1) and [`CITATION.cff`](CITATION.cff).

### Changed
- The `hcd` command needs an explicit `--config`; it never falls back to built-in settings.
- The install name is `hcd-priya`; the import name stays `hcd_analysis`.

## 0.1.0

The research code before it was packaged for release.
