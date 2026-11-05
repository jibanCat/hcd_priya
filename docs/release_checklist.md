# Release checklist (maintainers)

A release is a tagged version of the code. Releasing trained models or data is a separate decision, taken after the
code release (see the last section).

## 1. Version and records
- [ ] `hcd_analysis.__version__` is the new version; `pyproject.toml` reads it from there.
- [ ] `CHANGELOG.md` has a dated section for the version.
- [ ] `CITATION.cff` has the new `version` and a `date-released`.

## 2. Tests and continuous integration
- [ ] The supported CI job ("Python 3.11, pinned (supported)") passes on the release commit on GitHub.
- [ ] The newer-dependency jobs are read. A failure there is listed in the release notes and is never hidden.
- [ ] The older-libhdf5 job passes.
- [ ] The documentation builds on Read the Docs, with warnings treated as errors.

## 3. Distribution
- [ ] The sdist and wheel are built from a clean checkout of the release commit, with `SOURCE_DATE_EPOCH` set to the
      commit time.
- [ ] `python tools/ci/check_dist.py dist` reports 0 problems.
- [ ] The wheel installs into a fresh environment with the pinned constraints, and `hcd --help` runs.

## 4. Public content
- [ ] The diff since the last release holds no private filesystem path, no pointer into private notes, no credential,
      no analysis-derived number from unpublished work and no real-data result that is not yet public.

## 5. Review
- [ ] Every change since the last release was reviewed by a human maintainer. AI-assisted reviews are advisory and
      are labelled as such.
- [ ] Generative-AI use is recorded for the release notes and for any paper.

## 6. Tag and archive
- [ ] The maintainer approves, then tags the release commit and publishes the GitHub release.
- [ ] The release is archived (for example on Zenodo), and the archive DOI is added to `CITATION.cff`.

## 7. Models and data (a separate decision)
- [ ] A model is released only after its own acceptance: a clean lineage, a published manifest digest and a model
      card.
- [ ] The registry gets the model's entry only after that.
