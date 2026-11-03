# Release format for emulator bundles: HDF5 + JSON (`hcd_priya.h5/2`)

Status: the format for released emulator bundles, adopted subject to an independent review of its loader and a
compatibility test with an older HDF5 library (passed locally with libhdf5 1.10.8 reading a bundle written with 2.0.0;
not yet run in continuous integration). The trained models themselves are not distributed yet. The training
checkpoints (`.eqx` + `.norm.pkl` + `.meta.json`, written by `train.save_checkpoint`) remain the working format; a
release converts them once (`scripts/export_safe_bundle.py`). `hcd_priya.h5/2` adds to the earlier `hcd_priya.h5/1`
a lineage block and the map from absorber incidence (dN/dX) to the class weights w_c; an `h5/1` bundle loads only
through the separately named historical loaders (below). The earlier safetensors-layout prototype
(`docs/release_format.md`) is kept as evidence only and is not a second release format.

## Why

`.norm.pkl` is a pickle: loading it can run arbitrary code. A release needs files that load without executing anything
from the file, whose arrays carry names a loader can check one by one, and that standard tools can open. HDF5 is
already a dependency of this package and is readable everywhere (`h5ls`, `h5dump`, HDFView, any language); the JSON
sidecar is plain text.

## Layout

One member `<prefix>`:

- `<prefix>.h5`: the root attribute `format = "hcd_priya.h5/2"` and nothing else at the root; the group `/model` holds
  one dataset per network leaf, named by its JAX tree path (for example `.enc.layers[0].weight`); the group `/norm`
  holds the output normalizer as nested groups (an integer dictionary key is written `#<int>`). Every dataset is
  contiguous, allocated, unfiltered, attribute-free, little-endian float64.
- `<prefix>.json`: the checkpoint meta (schema 2.0: architecture, seed, k grid, the training cache's sha256 and the
  digest of its lineage record, the full code commit and `code_dirty: false`, the cache's file name, never a path),
  the tensor index (name, shape, dtype of every dataset) and the sha256 of the `.h5` file.

A bundle is a directory holding the members, `wc_map.json` (the map from Head A's dN/dX to the class fractions w_c,
refitted on the training cache by `scripts/refit_wc_map.py`, with its own lineage record; format
`hcd_priya.wc_map/1`, `hcd_analysis/emulator/wc_map.py`) and `manifest.json`, which lists every other file with its
sha256 and holds nothing else. A published bundle comes with the sha256 of its `manifest.json`.

### Keys and names

- **Sidecar** `<prefix>.json` (UTF-8 JSON, finite numbers only, no duplicate keys): `format` (`"hcd_priya.h5/2"`),
  `meta` (the schema-2.0 checkpoint meta), `tensors` (`{name: {"shape": [...], "dtype": "F64"}}`), `h5_sha256` (hex
  sha256 of the `.h5` bytes) and `h5_file` (the `.h5` file name, `<prefix basename>.h5`).
- **Manifest** `manifest.json`: exactly the keys `format` (`"hcd_priya.h5/2"`), `members` (member names in order),
  `files` (`[{"path", "sha256", "bytes"}]`, exactly the members' `.h5` and `.json` and the w_c map), `wc_map` (the map's
  file name, `"wc_map.json"`), `lineage` (the lineage block, below) and `provenance` (a JSON object, otherwise
  free-form; a production export adds `eligibility_checks`, a research bundle carries `research`: see "Production and
  research bundles" below).
- **Lineage block** `lineage` (format `hcd_priya.bundle_lineage/1`, written by `safe_h5.make_lineage_block`):
  - `cache`: the training cache's `sha256` and `size`, `lineage_record_sha256` (the sha256 of its canonical lineage
    record), `kind` and `kind_version`, `release_eligible`, its `settings`, `n_groups`, `rows` (`built` and
    `carried_validated` group counts) and `carried` (null, or `label: "carried_validated"`, the carried `fields`,
    the carry's `source_sha256` and `source_size`, `n_groups`, `validation_all_passed`, `validation_summary_sha256`
    and `promoted`: the fresh-row comparison that passed);
  - `catalogues`: `n` and `sha256` of the cache record's catalogue inputs (the sha256 of the canonical JSON of the
    sorted `[role, sha256]` pairs) and that `rule`;
  - `finder_ids` (the catalogue finder ids of the cache's lineage);
  - `code_commit` (the 40-hex commit every member was trained at) and `export_code` (`{commit, dirty}` of the export);
  - `wc_map`: `path`, `sha256`, `lineage_record_sha256`, `release_eligible` and the `cache_sha256` it was fitted to;
  - `n_members`, `release_eligible` and `ineligible_reasons` (computed when the block is made, never passed in; each
    reason is `"[CODE] text"` with a stable code: `LIN_*` for the block, `TIE_*` for the members' ties, the production
    rule's `RECIPE_NOT_DICT`, `ALL_SIMS_NOT_TRUE`, `RSV_KEY:<path>`, `RECIPE_RESEARCH`, `CV_SELECTION`, and
    `RESEARCH_BUNDLE` for a research bundle).
- **Tensor names.** A network leaf named `model<path>` in the index is the dataset `/model/<path>` (the path keeps
  its leading dot, for example index `model.enc.layers[0].weight` is `/model/.enc.layers[0].weight`). A normalizer
  entry `norm/<a>/<b>/...` is the dataset `/norm/<a>/<b>/...`, one group per path segment.

## Loading

```python
import hcd_analysis.emulator                     # enables JAX float64 on import (the format stores float64 only)
from hcd_analysis.emulator import safe_h5
from hcd_analysis.emulator.predict import predict_P_filt

ensemble, metas, norm = safe_h5.load_ensemble("path/to/bundle",
                                              expected_manifest_sha256="<the published manifest sha256>")
# theta9_unit, z_unit, tau0: the emulator inputs, built as in the README Quickstart
P_filt = predict_P_filt(ensemble, theta9_unit, z_unit, tau0, norm["P_filt"])   # (4, K): clean, LLS, subDLA, DLA
```

The public predictor (`hcd_analysis/predictor.py`) wraps this:

```python
import hcd_analysis.predictor as hp
emu = hp.load_bundle("path/to/bundle", "<the published manifest sha256>")   # None is refused
pred = emu.predict(params, z, tau_eff=tau_eff)   # class spectra, templates, w_c from the bundled map,
                                                  # the members' spread (a seed-spread diagnostic), provenance
P = hp.add_hcd(pred, A)                           # free amplitudes; only A_c >= 0 and 1 - sum_c A_c >= 0 imposed
```

`load_ensemble` is the release entry point. With the published digest it requires format `h5/2` and a lineage block
that establishes a clean lineage: every finder id in the release allow-list, an eligible training cache (its carried
rows labelled and promoted), the members' commit known and the export's tree clean, the w_c map eligible and fitted to
the block's cache and equal to the listed file, every member trained on the block's cache and lineage record at the
block's commit from a clean tree, and every member a production member (below); otherwise it refuses the bundle. A
research bundle is refused here (`load_research` reads it). It also refuses members that do not share one
architecture, k grid, parameter box and normalizer. `load_bundle` returns the members after the same checks;
`load_release` (digest required) also returns the parsed w_c map and the block. `load_member(prefix)` loads one member,
anchored only by its own sidecar unless the manifest's digests are passed (`expected_sha256`, `expected_json_sha256`);
a release therefore loads through `load_ensemble`. `verify_manifest(bundle_dir)` checks a bundle without loading it.
The `expected_manifest_sha256` keyword is required: pass the published digest. Passing `None` loads an unanchored
development bundle: the lineage is not enforced, an `UnanchoredBundleWarning` is issued and the returned metas say so
(`metas.anchored` is False, `metas.kind` is `"dev"`). The block is a declaration pinned by the manifest digest: each
product's builder verified its own inputs when it ran, and the loader checks that the declaration is complete,
consistent and eligible. Every loaded object says how it was loaded: `kind` (`"release"`, `"research"`, `"dev"` or
`"historical"`) and `release_verified` (True only after a release load; the fail-closed default is False), on
`ReleaseBundle`, `ResearchBundle` and `MemberMetas`; a slice of the metas is a plain list.

**Historical bundles.** An `h5/1` bundle (written before bundles carried lineage) is refused by the release loaders;
`load_ensemble_historical(bundle_dir, expected_manifest_sha256=..., reason=...)` and `load_bundle_historical` read it
with every `h5/1` format check and no lineage, log the reason, and mark the metas (`metas.historical_reason`,
`metas.kind == "historical"`).

## Production and research bundles

A **production** bundle holds members trained on every simulation of their training cache; it is what
`predictor.load_bundle` serves. A **research** bundle holds cross-validation fold members, or full-simulation research
models; it is never a production model, whatever its split. The rules live in one module,
`hcd_analysis/emulator/eligibility.py`; a step-by-step user guide with executable examples is
[`docs/tutorials/production_and_research_bundles.md`](tutorials/production_and_research_bundles.md).

**The loaders.**

| loader | takes | returns | refuses |
|---|---|---|---|
| `predictor.load_bundle(dir, sha256)` | a production bundle | `Bundle` (it recomputes the eligibility itself) | research, fold, ineligible, `h5/1`, `None` |
| `safe_h5.load_release(dir, expected_manifest_sha256=...)` (and `load_ensemble`, `load_bundle` with a digest) | a production bundle | `ReleaseBundle`, `kind` "release", `release_verified` True | the same |
| `safe_h5.load_research(dir, expected_manifest_sha256=..., purpose=...)`, `predictor.load_research_bundle(dir, sha256, purpose=...)` | a research bundle | `ResearchBundle` / `ResearchPredictor` (never a `ReleaseBundle` or `Bundle`; every prediction flagged `research_model`) | a production bundle, an empty purpose, no digest |
| `safe_h5.load_ensemble(dir, expected_manifest_sha256=None)` | any h5/2 bundle (development) | the ensemble, metas `kind` "dev"; a warning (naming a research bundle as such) | an `h5/1` bundle |

**The production metadata rule** (every release load, and `predictor.Bundle` again): each member's recipe is a JSON
object, declares `all_sims: true`, carries no research marking (`recipe.research`), and no key anywhere in the meta
normalises (lower case, every character but a-z and 0-9 removed) to a reserved held-out or fold name
(`reserved_keys/3`: holdout, holdoutsim(s), holdoutsimulations, heldout, heldoutsim(s), heldoutsimulations, fold(s),
foldid, foldidx, foldindex, foldnum, foldnumber, foldssha256, nheld, nheldrows, nholdout, leftout, leftoutsims,
withheld(sims), exclude(d)sims, testsims, loo, loocv, loso, kfold, cvfold; dicts and lists recursed; a null value
counts as absent; a recipe key that normalises to `research` is a research marking). Choices derived
from a cross-validation may be recorded in `recipe.cv_selection`, the closed schema `cv_selection/1`
(`{schema, source_cv_set_id, source_folds_digest, [source_record_sha256], derived: {knob: {value, rule}}}`, each value a
JSON scalar equal to the recipe knob it names); the scan exempts nothing. The bundle's w_c map must count every
snapshot group of the training cache (`WCMAP_NOT_FULL_CACHE` otherwise), so a fold map never passes as production.
Before a member's metas are read, these rules check what the bundle DECLARES; the content rules below verify it
against the cache.

**The content rules** (they need the training cache, so they run at export and acceptance, not at load):
`eligibility.production_content_check` ties the cache first, then recomputes the split from that cache's own rows with
the trainer's rule (`production_row_split/1`; the recipe must state `n_train` and `n_val`; "complete" means every
simulation of that cache, whatever their number), refits every member's normaliser with `train._fit_norm` on the
training rows (`norm_fingerprint/1`, every tensor compared bitwise first) and refits the w_c map on the cache
(`wcmap_refit/1`: the whole payload and the record's inputs, settings, finder ids, product, kind and validation entry;
never the writing run's code, time or settings digest). The record's input entry includes the cache's FILE NAME, so
the cache given to the export or the acceptance (`--cache`) must carry the name the map's record names: a renamed
copy of the right cache gives MISMATCH, and there is no override (rename the file back, or refit the map). Each component is PASS, MISMATCH or UNAVAILABLE: a difference within `DRIFT_BOUND`
= 1e-12 relative (per-tensor scale floor) is UNAVAILABLE only when a training environment is recorded and differs from
the current one; with none recorded the strict rule makes it MISMATCH. There is no override. `scripts/export_safe_bundle.py`
requires PASS everywhere before writing, checks again on the files read back, writes the manifest last and records the
outcomes in `provenance.eligibility_checks` (the loaders do not require that field; it is information).
`scripts/accept_retrained_ensemble.py` (report format `hcd_priya.ensemble_acceptance/2`) adds the checks
`training_set_declared_complete`, `split_recomputed_equal`, `normaliser_fingerprint_equal`, `wc_map_refit_equal` and
records `content_check_environment` and `content_checks`.

**Research bundles.** The marking is bundle-level: `provenance.research` (schema `research/1`: `purpose`, `split_rule`
one of `production_row_split/1`, `multi_holdout_row_split/1` or `unverifiable`, the declared `split`, `folds_digest`,
optional `cv_set_id`, `fold_file_sha256` and `restored_epoch`, the `content_checks` and `wc_map` = `{rule,
fitted_on_sims_sha256, n_sims}`), and the block's reason `[RESEARCH_BUNDLE] research bundle: <purpose>`
(`make_lineage_block(research=...)`); `write_manifest` refuses a block and provenance that disagree. A member that
declares an exclusion under the declared rule is admitted as it is (the existing cross-validation members); a member
that declares no exclusion must carry `recipe.research` naming the purpose; one split per bundle. The research loader
enforces the lineage and the ties, refuses a stale block (lineage or tie codes present, the research reason absent;
the per-member production codes are not compared, so a later reserved name does not orphan a bundle) and checks the
map's simulation count. `scripts/export_research_bundle.py` writes research bundles: the split is declared on the
command line and checked against every member; under `production_row_split/1` and `multi_holdout_row_split/1` it is
also recomputed from the cache. Under `unverifiable` the split and the normaliser are NOT recomputed (recorded
UNAVAILABLE): the declared held-out set must then be non-empty whenever a member declares an exclusion, and must equal
any member's `holdout_sims` / `holdout_sim` exactly, because the w_c map is fitted on the cache minus that set. The w_c
map is refitted on the training simulations only (`wc_map.refit_subset`, rule `wcmap_refit_subset/1`;
byte-identical to `wc_map.refit` for the full set). Research
bundles need a release-grade cache and finder ids; research on other caches stays on the training checkpoints.

**Historical synthetic bundles.** Synthetic `h5/2` bundles written before this rule (recipe `{"synthetic": true}`,
no `all_sims`) are historical artefacts: the release loaders refuse them (`ALL_SIMS_NOT_TRUE`) and the development path
loads them. Synthetic bundles written now declare `all_sims: true`.

**What protects what.**

| layer | protects against | does not protect against |
|---|---|---|
| the published manifest digest, pinned after a recorded acceptance | substitution, corruption, an unaccepted bundle presented as accepted | a compromised maintainer, an unprotected registry manifest |
| the loader metadata rules and the separate types | accidental export or load of fold or research models as production; mislabelled recipes using reserved names | deliberately forged metadata with a recomputed digest; a novel key spelling outside the reserved list |
| the content checks at export and acceptance | a split, normaliser or w_c map that is not from the declared training set | a forger who has the cache and refits; network weights trained on other rows than the declared ones |
| out of scope | in-process objects (for a hand-built `ReleaseBundle`, `Bundle` recomputes the block, the metas' rules and marks, the map rule and the map's identity, but not the network arrays or the normaliser), pickled objects, the training checkpoints, the frozen benchmark environment | |

`ResearchPredictor` has the same prediction interface as `Bundle`, so code that accepts any predictor object does not
tell them apart by type; such code is protected by what it pins, not by the type. A fold-refit w_c map file is not
self-describing outside its bundle: its record names the full cache, and it differs from the full map in
`delta_c.n_blocks`, `n_sims`, `xbar.n_groups`, the residual statistics and the record's `product`. Every production
loader and `Bundle` refuse one placed in a production bundle (`WCMAP_NOT_FULL_CACHE`: its `n_blocks` is not the
block's `cache.n_groups`; fail-closed edge: a full cache with a snapshot group without sightlines would be refused
too), and the production export's map refit compares the whole map. A full-simulation research model trained
without `recipe.research` looks like a production member to every rule.

## Compatibility with earlier versions of the library

The production and research rules apply from the D5 commits of branch `a6-d5-fold-guards` (first commit `0b6fe33`,
package version 0.1.0; not yet released). The library at earlier commits (for example `372204f` and the frozen
benchmark's `3496030`) predates them. Differences a user of an earlier version meets:

- synthetic `h5/2` bundles written before the rule (recipe without `all_sims`) are refused by the release loaders
  (`ALL_SIMS_NOT_TRUE`) and load through the development path; synthetic bundles written now declare `all_sims: true`
  and their made-up w_c map counts the synthetic cache's groups, so their digests differ from earlier ones;
- a bundle whose w_c map does not count every group of its training cache is refused (`WCMAP_NOT_FULL_CACHE`);
- `ReleaseBundle` has 11 fields (`kind` mandatory, `release_verified` defaulting to False); `MemberMetas(...)` needs
  `kind=`; a slice of the metas is a plain list; `ResearchBundle` is a separate type;
- `predictor.Bundle` takes only a release-verified `ReleaseBundle` with its release-verified metas;
- `metas.anchored and metas.lineage_verified` is no longer a production test (both are true after a research load):
  use `metas.release_verified` (or `kind == "release"`);
- `ineligible_reasons` strings are `"[CODE] text"`;
- `provenance` must be a JSON object on every write and every load, the development path included;
  `write_manifest_historical` refuses a research marking;
- a digest given to `load_ensemble` / `load_bundle` that is not 64 lowercase hex digits (and not None) is refused
  before the manifest is read;
- `scripts/accept_retrained_ensemble.py` requires `--wc-map` and `--wc-map-sha256`, writes report format
  `hcd_priya.ensemble_acceptance/2`, and its `trained_on_every_simulation` takes a member meta (a recipe raises
  `TypeError`);
- `scripts/export_safe_bundle.py` requires the recipe to state `n_train` and `n_val` and the content checks to PASS.

## What the loader guarantees

- **No code execution from the files.** Nothing is unpickled or evaluated; the network is built from code
  (`model.Emulator(**arch)` under abstract evaluation) and the arrays are matched to it by name, shape and dtype.
- **Bounded reads.** Every file is opened without following a symlink and must be a regular file. A file read into
  memory has a size bound checked on its descriptor before it is read (`safe_io.MAX_JSON_BYTES` for a sidecar or
  manifest; the dataset bound plus metadata for an `.h5`); the manifest check compares each listed file's size
  before hashing it.
- **Digest before parse.** The `.h5` bytes are read once and their sha256 compared with the sidecar and with the
  manifest digest before the HDF5 library parses them; the same in-memory bytes are then opened.
- **A strict structure.** Refused: soft and external links, objects reachable twice, named datatypes, external,
  virtual, compact, chunked or filtered storage, storage not allocated inside the file, attributes other than the root
  `format` (a scalar string, whose shape and type are checked before its value is read), a user block, anything
  outside `/model` and `/norm`, a group inside `/model`, a group without datasets inside `/norm`, a link name longer
  than 255 characters, non-finite values, more objects, depth or bytes than the bounds in `safe_h5.py`. Shapes of the
  network and of the normalizer are checked from metadata before any data is read. Any HDF5 error becomes a
  `SafeFormatError`.
- **Never a silent down-cast.** A process without JAX float64 is refused.

## Writing

`safe_h5.export_member(prefix, model, meta, norm)` writes one member and never overwrites: each file is written through
the descriptor of a new temporary file and published by hard link only if that name still holds it. Files are
byte-identical across exports for a fixed HDF5 library version (no timestamps are stored). The writer limits the HDF5
object versions to those libhdf5 1.10 reads (`libver=("earliest", "v110")`). The sidecar's `h5_sha256` is computed
through the same descriptor that wrote the file, never by reopening its name. `export_wc_map` copies the w_c map in;
`write_manifest(bundle_dir, prefixes, provenance, lineage=block)` writes the manifest, last. The release export
(`scripts/export_safe_bundle.py`) refuses ineligible inputs before writing anything; the research export
(`scripts/export_research_bundle.py`) writes research bundles. Synthetic, self-consistent `h5/2` bundles for tests and
CI: `hcd_analysis.emulator.release_testing.write_synthetic_bundle` and `write_synthetic_research_bundle`.

## Inspecting a member with standard tools

```
h5ls -r member0.h5              # every dataset and its shape
python -c "import json; print(json.load(open('member0.json'))['meta']['arch_cfg'])"
```
