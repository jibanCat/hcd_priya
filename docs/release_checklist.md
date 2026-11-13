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
        that list must be empty at a release. It is empty now. Its last entry, a private path in the package-data
        file `hcd_analysis/emulator/hcd_lit_dndx_corrected.json` (`cache_provenance.path`), was fixed by rewriting
        that path to the repository-relative one; no analysis-lock re-cut was needed, and the historical lock is
        unchanged. The release-mode scan above remains blocking for every release.
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

## 7. Models (a separate decision, with its own approval)

The trained models are published as the package `hcd-priya-models` (`model_package/`), which `hcd-priya` requires.
The order matters, because the code names the model it accepts:

1. the model is accepted, and its entry (version and the sha256 of its `release.json`) is added to
   `hcd_analysis/published_models.py`;
2. the code is merged and tagged **with that entry** (code tagged without it accepts no model);
3. the model package is built from the same accepted release and published;
4. then the code package; then the archive.

Each step needs the maintainer's approval. The publishing workflow (`.github/workflows/publish.yml`) is manual and
disabled until then.

- [ ] A model is released only after its own acceptance: a clean lineage, a published manifest digest and a model
      card.
- [ ] The model package is prepared from the accepted release, in a clean clone: `python model_package/prepare.py
      --release RELEASE_DIR --release-sha256 SHA256` checks the digest, copies only what the release names and its
      documentation files, and verifies every file. Its version is the release's version. The model files are not
      committed.
- [ ] The entry that `prepare.py` prints (name, version, format, the sha256 of `release.json`) is in
      `hcd_analysis/published_models.py` of the commit to be tagged, and the dependency range of
      `hcd-priya-models` in `pyproject.toml` contains that version.
- [ ] The model wheel and source archive are built (`python -m build --outdir <folder> model_package`), and
      `python tools/ci/check_dist_content.py <folder> --release` reports 0 findings.
- [ ] **Blocking: the fresh install.** In a new environment that holds only the two built wheels (no clone on the
      import path), with `HCD_PRIYA_ALLOW_SYNTHETIC` unset: `python -I tools/ci/readme_first_example.py README.md`
      passes (the first example of the README, as written), and `python -I -m hcd_analysis.check_release --published
      --reference reference_predictions.json` ends in the plain line `PASSED` (every file verifies; the
      predictions agree with the reference predictions published beside the release). A run that ends in
      "PASSED (synthetic build ...)" is not this check. The reference is written once, on the supported
      environment, with `--write-reference`.
- [ ] A dry run on the test index, then the model package on the index, then the code package, by the manual
      workflow. A synthetic build (version `0+synthetic`) cannot be uploaded: the index refuses a local version.
- [ ] The images of `README.md` are relative links, and the README is the package's long description: before the
      upload they point to absolute URLs at the release tag, or the package index page shows them as broken.
- [ ] After publication: the statements "not published yet" in `README.md`, `model_package/README.md` and the
      docstrings are updated, and the install commands become `pip install hcd-priya`.
