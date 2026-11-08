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
- [ ] The bit-level digest replay passes with the supported pinned stack on the kind of CPU the digests were
      recorded on, x86-64 with AVX-512 (CI never runs it: CI compares floating-point results with stored references
      at stated tolerances). A digest failure on another kind of CPU is inconclusive: rerun on such a machine, and
      never make it pass by re-recording the stored digests (`tests/portable_numerics.py` explains why). From the
      clone, with an editable install, one file per run:
      `HCD_EXACT_REPLAY=1 HCD_PINNED_STACK=1 JAX_PLATFORMS=cpu python -m pytest -q tests/<file>` for
      `test_eligibility.py`, `test_gateE_invariants.py` and `test_wc_map.py`. Run pytest directly:
      `tools/ci/run_test_tier.py` declares these digest tests as skipped by design and fails a file where they run.

## 3. Distribution
- [ ] The sdist and wheel are built from a clean checkout of the release commit, with `SOURCE_DATE_EPOCH` set to the
      commit time.
- [ ] `python tools/ci/check_dist.py dist` reports 0 problems.
- [ ] **Blocking:** `python tools/ci/check_dist_content.py dist --release` reports 0 findings. It scans every file,
      file name, link target and comment of the built wheel and sdist, and the sdist's tar owner fields, for private
      filesystem paths, pointers into private notes, batch-system account directives, credentials, confidentiality
      markers, undeclared e-mail addresses and a builder's account name, and prints where, never what. No wheel or
      sdist is published before it passes.
      - Build the release archives on a CI runner, or re-pack the sdist with anonymous owners: an sdist built on a
        personal account names that account in every tar header (rule archive-owner).
      - An e-mail address in shipped text is declared in `pyproject.toml` as an author, written with a reserved
        example domain, or removed.
      - CI runs the same scan against `tools/ci/dist_content_known.txt`, the known findings still to be cleaned
        (listed by rule, file and line digest, one line per occurrence), so that CI fails on any new or stale one;
        that list must be empty at a release. One of them, a path in the package-data file
        `hcd_analysis/emulator/hcd_lit_dndx_corrected.json`, can only be fixed together with a re-cut of the
        analysis lock, which needs the maintainer's separate approval.
- [ ] The wheel installs into a fresh environment with the pinned constraints, and `hcd --help` runs.

## 4. Public content
- [ ] The diff since the last release holds no private filesystem path, no pointer into private notes, no credential,
      no analysis-derived number from unpublished work and no real-data result that is not yet public.
- [ ] A maintainer has read what the distribution ships for unpublished numbers and results, which no automated scan
      recognises (the content scan of section 3 finds paths, pointers, credentials and markers only).

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
