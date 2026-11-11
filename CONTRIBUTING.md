# Contributing to hcd_priya

Thank you for your interest. This page says how to report a problem, how to ask for help and how to contribute
changes. The project is research software under active development; the supported environment and what the tests
cover are described in [`docs/compatibility.md`](docs/compatibility.md).

## Reporting an issue

Open an issue on the repository's GitHub issue tracker. Please include:

- what you ran (the command or a short script) and what you expected to happen;
- what happened instead: the full error message or traceback;
- the package version (`python -c "import hcd_analysis; print(hcd_analysis.__version__)"`), your Python version and
  operating system, and whether you installed the supported environment (`pip freeze` output helps);
- the smallest example that shows the problem, if you can make one with synthetic inputs.

Please do not attach private data, unpublished results or model files to an issue. If a problem only shows up with
data you cannot share, describe the data's shape and type instead.

## Asking for help

Questions about installing or using the package are welcome as issues too; say in the title that it is a question.

## Contributing changes

1. **Discuss first.** For anything beyond a small fix, open an issue describing the change before writing it, so we
   can agree on the approach.
2. **Set up the supported environment** from a clone (Linux, CPython 3.11):

   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -c tools/ci/constraints-devenv.txt -e ".[test]"
   ```

   The install is editable, so the code you change is the code that runs. The pipeline records the clone's commit,
   which describes the code only once your changes are committed; with uncommitted changes, each snapshot's
   metadata says so (`code_dirty`).

3. **Work on a branch** of your fork, and keep each pull request to one change.
4. **Test.** Add tests for what you change. Tests use synthetic inputs only: no private data, unpublished results or
   values derived from them. Run each affected test file in its own process:

   ```bash
   JAX_PLATFORMS=cpu python -m pytest -q tests/test_<name>.py
   ```

   A new test file must also be listed in [`tests/ci_tiers.toml`](tests/ci_tiers.toml), with a reason if it cannot
   run in continuous integration. To run what CI runs from a clone:

   ```bash
   JAX_PLATFORMS=cpu python tools/ci/run_test_tier.py --tier source --out <new_results_dir>
   ```

5. **Open a pull request** against `main`. Describe what changed and why, and which tests cover it. The CI job for the
   supported environment must pass; the jobs for newer environments are informational, but a change should not add
   new failures to them.
6. **Say if you used generative AI.** If AI tools helped write the code, tests or documentation, name them and say
   what they did in the pull request description.
7. **Review.** Every pull request is reviewed by a human maintainer before it is merged.

Please match the style of the surrounding code (names and comments), and keep large data and model files out of
the repository.

**Code format.** The Python code is formatted with [black](https://black.readthedocs.io), line length 100, with the
version pinned in `tools/ci/requirements-format.txt` (settings in `pyproject.toml`, `[tool.black]`). Before you
commit, run:

```bash
pip install -r tools/ci/requirements-format.txt
black .
```

The CI job "Format" runs `black --check .` and fails if any file would change. One file is excluded and never
reformatted: `scripts/gen_analysis_lock.py`, the generator of the historical analysis lock.

**What a reformat changes.** A format-only commit changes source text, not behaviour. Two provenance mechanisms
follow the source text, so they see a reformat:

- **Catalogue lineage records** name the finder's source by digest: the sha256 of `hcd_analysis/catalog.py` and
  `hcd_analysis/voigt_utils.py`, and a digest of the N_HI estimator's source (`catalog._source_digests` and
  `catalog._estimator_source_sha256`). A catalogue built after the black format commit (`3f58b5e`) therefore
  records different finder source digests than an earlier catalogue built from identical code. The records of
  existing catalogues are kept as they are. The catalogue acceptance script
  (`scripts/accept_rebuilt_catalogues.py`) already requires one code commit and one set of finder digests across
  the groups it accepts, so catalogues built before and after a reformat are never accepted together.
- **The cache-carry tool** (`scripts/carry_validated_cache.py`) compares the row-path code at the builder commits
  of record with the code at HEAD, as text. At a head after the format commit, that comparison shows
  formatting-only differences. The rule is unchanged: such a carry still needs the diff reviewed and its sha256
  passed with `--accept-row-path-diff`. No diff is accepted in advance.
