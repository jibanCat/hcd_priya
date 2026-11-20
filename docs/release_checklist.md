# Release checklist (maintainers)

A release is three things with one version: the code package `hcd-priya`, the model package `hcd-priya-models`, and
the model record (three files on the GitHub release). The code names the model it accepts, so the order is fixed:
the model is accepted and named in the code; the code is merged and tagged; the model package is published; then
the code package. Every step that publishes waits for the maintainer's approval, and an upload to the package index
cannot be replaced or withdrawn.

Sections 1 to 6 are what must be true of the release commit. Section 7 is the order of publication.

## 1. The release commit: version, date and texts
- [ ] `hcd_analysis.__version__` is the new version; `pyproject.toml` reads it from there.
- [ ] The release date is the day of publication. `CHANGELOG.md` has a section for the version whose heading holds
      the placeholder `RELEASE_DATE`, and `CITATION.cff` holds the commented line `# date-released: RELEASE_DATE`.
      In the last commit before the tag, the heading takes the day and the commented line becomes the real line
      `date-released: YYYY-MM-DD` with the same day. A test keeps the two files together, and the publishing
      workflow refuses an upload to the index while the placeholder stands.
- [ ] `CITATION.cff` has the new `version`.
- [ ] The README is the page of the code package on the index, which shows a relative link or image as broken and
      cannot be amended. In the release commit every link and image of `README.md` into this repository is an
      absolute URL at the release tag (a test checks it, and that each target exists); links inside the page stay.
- [ ] The texts describe the state after publication. No sentence of `README.md`, `model_package/README.md`
      (the page of the model package on the index), `docs/`, `examples/mock_recovery.py` or a docstring says that
      the packages or the model are not published yet; the install command is `pip install hcd-priya`; the model
      record is named by the URL of the GitHub release of the version. This is done before the tag, not after the
      upload.
- [ ] The accepted model's entry (name, version, format, the sha256 of its `release.json`, as `prepare.py` prints
      it) is in `hcd_analysis/published_models.py`, and `pyproject.toml` pins `hcd-priya-models` to exactly that
      version (`==`). Code tagged without the entry accepts no model.

## 2. Tests and continuous integration
- [ ] The two supported CI jobs ("Python 3.11, pinned (supported), wheel tier" and "Python 3.11, pinned (supported),
      source tier") pass on the release commit on GitHub.
- [ ] The latest nightly run on the release commit passed (`.github/workflows/nightly.yml`, all three stacks), and the
      release validation workflow (`.github/workflows/release-validation.yml`, started by hand on that commit) passes:
      it refuses to go on otherwise. Nightly failures block a release. A failure of a newer-dependency stack is
      fixed, or listed in the release notes and never hidden, before a passing nightly run is made on the commit. A
      commit that no daily run tested (a commit of another branch, or any commit before the daily schedule exists on
      `main`) needs a nightly run started on it by hand first.
- [ ] The release-level tests (`tests/ci_tiers.toml`) ran with the real assets on the release commit (`python
      tools/ci/run_test_tier.py --tier assets`), and the record of that run is attached to the release.
      The science tests of the model need `HCD_SCIENCE_MODEL_CONFIG` (a JSON naming the release to be published by
      its directory and anchor), `HCD_SCIENCE_GOLDEN` (a golden file made from that release with
      `tools/ci/make_science_golden.py`) and `HCD_PINNED_STACK=1`; the training-cache test needs
      `HCD_SCIENCE_LF_CACHE` and `HCD_SCIENCE_LF_CACHE_SHA256`.
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
- [ ] The wheel installs into a fresh environment with the pinned constraints, together with the model wheel
      (`hcd-priya` pins `hcd-priya-models` exactly, so the code wheel alone does not install before the model
      package is on the index), and `hcd --help` runs.

## 4. Public content
- [ ] The diff since the last release holds no private filesystem path, no pointer into private notes, no credential,
      no analysis-derived number from unpublished work and no real-data result that is not yet public.
- [ ] A maintainer has read what the distribution ships for unpublished numbers and results, which no automated scan
      recognises (the content scan of section 3 finds paths, pointers, credentials and markers only).

## 5. Review
- [ ] Every change since the last release was reviewed by a human maintainer. AI-assisted reviews are advisory and
      are labelled as such.
- [ ] Generative-AI use is recorded for the release notes and for any paper.

## 6. The model release and its record
- [ ] A model is released only after its own acceptance: a clean lineage, a published manifest digest and a model
      card.
- [ ] The model package is prepared from the accepted release, in a clean clone: `python model_package/prepare.py
      --release RELEASE_DIR --release-sha256 SHA256 --allow-extra examples` checks the digest, copies only what the
      release names, its documentation files and its `examples` folder, and verifies every file. The release holds
      an `examples` folder that `release.json` does not name, so the command names it; without the option
      `prepare.py` stops. Its version is the release's version. The model files are not committed.
- [ ] The model wheel and source archive are built (`python -m build --outdir <folder> model_package`), and
      `python tools/ci/check_dist_content.py <folder> --release` reports 0 findings.
- [ ] **Blocking: the fresh install.** In a new environment that holds only the two built wheels (no clone on the
      import path), with `HCD_PRIYA_ALLOW_SYNTHETIC` unset: `python -I tools/ci/readme_first_example.py README.md`
      passes (the first example of the README, as written), and `python -I -m hcd_analysis.check_release --published
      --reference reference_predictions.json` ends in the plain line `PASSED` (every file verifies; the
      predictions agree with the reference predictions published beside the release). A run that ends in
      "PASSED (synthetic build ...)" is not this check. The reference is written once, on the supported
      environment, with `--write-reference`.
- [ ] What the reference compares, and what it does not. The reference predictions hold four arrays: `k`,
      `P_class`, `w_c` and `dndx`. The CDDF, the held-out errors of dN/dX and the CDDF, and the validation fields of
      the class spectra are not in it. `check_release` checks their files (every file against its digest, and the
      P1D held-out record's tables against its residual vectors); their values are compared with stored golden
      values by the release-level science tests of section 2, which the publishing workflow does not run. So those
      tests are run on the release commit with the release's files before anything is published.
- [ ] The model record holds three files: the release archive, `reference_predictions.json` (with its sha256,
      which the publishing workflow checks before it runs `check_release --reference`), and the example download
      (one archive of the directory `examples/mock_recovery.py` reads; its `example_download.json` and `SHA256SUMS`
      verify, and its five sha256 equal `DOWNLOAD` in the script). The URL of the example download and the sha256
      of its archive are `DOWNLOAD_URL` and `DOWNLOAD_ARCHIVE_SHA256` in `examples/mock_recovery.py`, and
      `docs/likelihood.md` gives the same two.

## 7. Publishing, in this order

The publishing workflow (`.github/workflows/publish.yml`) starts by hand only. It builds one package per run and
uploads it by trusted publishing: the index trusts this workflow of this repository in a named environment, and no
password or token is stored. Each upload waits for the maintainer's approval in that environment.

**Once, before the first release.**
- [ ] On the index and on the test index (two sites, two accounts), a pending trusted publisher for each package:
      owner `jibanCat`, repository `hcd_priya`, workflow `publish.yml`, and the environment of the table below. The
      index allows one pending publisher per owner, repository, workflow and environment, so the two packages need
      different environments.

  | Package | Test index | Index |
  |---|---|---|
  | `hcd-priya-models` | `testpypi-models` | `pypi-models` |
  | `hcd-priya` | `testpypi` | `pypi` |

- [ ] The four environments exist in this repository, each with the maintainer as required reviewer; `pypi` and
      `pypi-models` accept the tags `v*` only.

**For each release.**
1. [ ] The release pull request is merged: its merge commit is the release commit (sections 1 to 6).
2. [ ] The nightly workflow is started by hand on the release commit and passes on its three stacks; then the
       release validation workflow, started by hand on the same commit, passes. The publishing workflow refuses to
       build a commit without a passed nightly run.
3. [ ] The maintainer creates the GitHub release of the version as a **pre-release**: tag `vX.Y.Z` on the release
       commit, with the three files of the model record attached. Their sizes and sha256 are checked against the
       acceptance record.
4. [ ] Rehearsal on the test index, model package: the publishing workflow is started from the tag with package
       `hcd-priya-models`, index `testpypi`, and the five inputs of the model record:
       `release_archive_url` and `reference_url` (the two files on the GitHub release:
       `https://github.com/jibanCat/hcd_priya/releases/download/vX.Y.Z/<file>`), `release_archive_sha256`,
       `reference_sha256`, and `release_sha256` (the sha256 of `release.json`, the one in
       `hcd_analysis/published_models.py`). The build job checks the digests, builds, scans, and runs the fresh
       install of section 6; the upload waits for approval in `testpypi-models`.
5. [ ] Rehearsal on the test index, code package: the same from the tag with package `hcd-priya`, index `testpypi`,
       the other inputs empty; approval in `testpypi`.
6. [ ] Both packages are installed from the test index into a clean environment (the dependencies from the index),
       and the README's first example and `python -I -m hcd_analysis.check_release --published` pass there.
7. [ ] **The upload to the index, which cannot be undone.** Step 4 with index `pypi` (approval in `pypi-models`),
       then step 5 with index `pypi` (approval in `pypi`). The model package comes first: the code package requires
       it. The workflow refuses unless it runs from the tag of the version and the release date is written. A
       synthetic build (version `X.Y.Z+synthetic`) cannot be uploaded: the index refuses a local version.
8. [ ] `pip install hcd-priya` in a clean environment installs both packages, and the README's first example runs.
       The maintainer then un-marks the pre-release and sets it as the latest release.
9. [ ] Later: the release is archived (for example on Zenodo), and the archive DOI is added to `CITATION.cff`.

If the rehearsal (steps 4 to 6) fails for a reason in the code, nothing is on the index yet: the pre-release and its
tag are deleted, the fix is merged, and the steps are repeated from step 1.
