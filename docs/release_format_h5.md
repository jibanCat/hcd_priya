# Release format for emulator bundles: HDF5 + JSON (`hcd_priya.h5/2`)

Status: the pickle-free format for single-fidelity emulator bundles (see Scope below; the multi-fidelity correction
is a separate record, "The multi-fidelity record" and "The release manifest" below), adopted subject to an independent review of
its loader and a compatibility test with an older HDF5 library. That test runs in continuous integration (job "HDF5
release format with an older libhdf5"): a synthetic two-member single-fidelity bundle (small untrained networks, a
made-up lineage block and w_c map) written with the pinned stack is read bit for bit with h5py built against
Ubuntu's libhdf5 1.10, and the loader's tests pass there. It covers the member files and the manifest of that
synthetic bundle, not a trained release bundle or the multi-fidelity record. The trained models themselves are not
distributed yet.
The training
checkpoints (`.eqx` + `.norm.pkl` + `.meta.json`, written by `train.save_checkpoint`) remain the working format; a
release converts them once (`scripts/export_safe_bundle.py`). `hcd_priya.h5/2` adds to the earlier `hcd_priya.h5/1`
a lineage block and the map from absorber incidence (dN/dX) to the class weights w_c; an `h5/1` bundle loads only
through the separately named historical loaders (below). The earlier safetensors-layout prototype
(`docs/release_format.md`) is kept as evidence only and is not a second release format.

## Scope: single fidelity

An `hcd_priya.h5/2` bundle holds the members of ONE training cache and the w_c map fitted to it. It has no slot for a
multi-fidelity (LF to HR) correction, and the predictor (`hcd_analysis/predictor.py`) applies none to a bundle loaded
alone: a bundle trained on the low-fidelity cache gives low-fidelity predictions, and every such prediction's labels
say so (`product` names the training cache's fidelity; `mf_correction` is `none`). The multi-fidelity correction is
kept outside the bundle, so the bundle's fixed manifest keys (below) and its digest do not change: it is a separate
record bound to one bundle ("The multi-fidelity record"), and an LF+MF release publishes the two under one release
manifest ("The release manifest"); an LF+MF prediction's `mf_correction` label says the correction is applied. An
`hcd_priya.h5/2` bundle alone is therefore not the complete release model.

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
# theta9_unit, z_unit, tau0: the emulator inputs (the nine parameters and the redshift scaled to the unit
# interval, and the mean-flux input), built as in hcd_analysis/emulator/README.md, section 6
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
counts as absent; a recipe key that normalises to `research` is a research marking). Nor may the meta name a
proper subset of the cache's simulations anywhere (`simulation_sets/1`, code `SIM_SET:<path>`). The simulations are
found by VALUE: a list or tuple of names, a dict keyed by them, one name, or names separated by spaces, commas,
semicolons or bars, at any depth and under any key. So `excluded: [...]`, `train_sims: [...]` or a list under an
arbitrary key is refused like a reserved name, and a list of every simulation is not an exclusion. Where the cache is
at hand (the block's writer `make_lineage_block`, the exports, the acceptance) a simulation name is one of the
cache's; a load has no cache, so it recognises the simulation-name form (the cache's folder naming,
`eligibility.SIM_NAME_FORM`), takes the simulation count from the w_c map, and refuses a block whose write-time verdict
is ineligible. Not recognisable from metadata: simulations given by index, by digest or in prose; for a production
member the content checks below still see training on fewer simulations (the normaliser recomputed on every
simulation's training rows differs). Choices derived
from a cross-validation may be recorded in `recipe.cv_selection`, the closed schema `cv_selection/1`
(`{schema, source_cv_set_id, source_folds_digest, [source_record_sha256], derived: {knob: {value, rule}}}`, each value a
JSON scalar equal to the recipe knob it names); the scan exempts nothing. The bundle's w_c map must count every
snapshot group of the training cache (`WCMAP_NOT_FULL_CACHE` otherwise), so a fold map never passes as production.
At load, these rules check what the bundle DECLARES; the content rules below verify it against the cache.

**The content rules** (they need the training cache, so they run at export and acceptance, not at load):
`eligibility.production_content_check` ties the cache first, then recomputes the split from that cache's own rows with
the trainer's rule (`production_row_split/1`; the recipe must state `n_train` and `n_val`; "complete" means every
simulation of that cache, whatever their number), refits every member's normaliser with `train._fit_norm` on the
training rows (`norm_fingerprint/1`, every tensor compared bitwise first) and refits the w_c map on the cache
(`wcmap_refit/1`: the whole payload and the record's inputs, settings, finder ids, product, kind and validation entry;
never the writing run's code, time or settings digest). The record's input entry includes the cache's FILE NAME, so
the cache given to the export or the acceptance (`--cache`) must carry the name the map's record names: a renamed
copy of the right cache gives MISMATCH, and there is no override (rename the file back, or refit the map). Each
component is PASS, MISMATCH or UNAVAILABLE: a difference within `DRIFT_BOUND` = 1e-12 relative (per-tensor scale floor)
is UNAVAILABLE only when a training environment is recorded and differs from the current one; with none recorded the
strict rule makes it MISMATCH. There is no override. `scripts/export_safe_bundle.py` requires PASS everywhere before
writing, checks again on the files read back, writes the manifest last and records the
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
also recomputed from the cache. Every member's excluded set is derived from every simulation set its metadata
names (`simulation_sets/1`, by value, under any key): a held-out list adds its names, a training list (`train_sims`
and its spellings) its complement, and a list of every simulation adds nothing. A list whose key says neither held
out nor trained on (`RES_SIM_SET_UNDECLARED`), a training list that is not the bundle's training set
(`RES_SIM_SET_TRAINING`) and an exclusion key that names no simulation (`RES_EXCLUSION_UNPARSED`) are refused. Under
`unverifiable` the split and the normaliser are NOT recomputed (recorded UNAVAILABLE): the declared held-out set must
then be non-empty whenever a member declares an exclusion, and must EQUAL every member's excluded set (hence their
union), because the w_c map is fitted on the cache minus that set; a member recording `holdout_sims: [a]` and
`withheld_sims: [b]` needs both `a` and `b` declared. Under the recognised rules the excluded set must equal the split
too. The block's writer judges the admission with the cache's names and records any refusal (a `RES_*` reason), which
the research loader enforces, and `scripts/export_research_bundle.py` also judges the members as the research loader
will, without the cache's names, and refuses before writing anything if that loader would refuse them. The w_c map is
refitted on the training simulations only (`wc_map.refit_subset`, rule `wcmap_refit_subset/1`; byte-identical to
`wc_map.refit` for the full set). Research bundles need a release-grade cache and finder ids; research on other caches
stays on the training checkpoints.

**The `simulation_sets/1` vocabulary.** Where a simulation set is found:

- as a list or tuple element, a dict key, a whole string, or a token of a string split on spaces, commas, semicolons or
  bars, at any depth;
- a string is a simulation name when it is one of the cache's names (when the cache is at hand: the block's writer, the
  exports, the acceptance), one of the declared held-out names (a research load), or in the simulation-name form
  (`eligibility.SIM_NAME_FORM`, the cache's folder naming);
- the value of a reserved held-out key (`eligibility.SET_FORMS`: holdoutsim(s), heldoutsim(s), heldout, holdout,
  holdoutsimulations, heldoutsimulations, leftout(sims), withheld(sims), exclude(d)sims, testsims) or of a training-set
  key (`eligibility.TRAIN_SET_FORMS`: trainsim(s), trainingsim(s), trainsimulations, trainingsimulations, trainedon,
  trainedonsims) is read as simulation names whatever their form; a value of such a key that is not a name or a list of
  names fails closed.

What a set means is read from the nearest dict key above it, normalised as for the reserved names (lower case, a-z
and 0-9 only):

- **trained on** (its complement is held out): the key contains `train`, for example `train_sims`,
  `training_simulations`, `trained_on`;
- **held out**: the key is a reserved held-out form or contains `holdout`, `heldout`, `withheld`, `exclud`, `leftout`,
  `omit` or `test`, for example `holdout_sims`, `excluded`, `left_out_runs`, `test_sims`;
- **no stated meaning**: any other key, or one that matches both, for example `val_sims`, `eval_sim`, `cache_sims`,
  `weights`. A research member with such a set is refused (`RES_SIM_SET_UNDECLARED`) unless the set lists every
  simulation; a production member is refused (`SIM_SET:<path>`) for any set that is not every simulation, whatever its
  meaning.

A substring such as `test` also matches words like `latest`; that can only refuse a set, never admit one. Not
recognised by the tokeniser: a name is read only as an exact whole token, so a name inside a path (`/data/<name>`),
with a suffix, in another case, in quotes or brackets (a stringified list), joined by `/` or `+`, or several names in
one list element is not read, at write or at load (a reserved held-out or training-set key still reads its value, and
for a production member the content checks still see training on fewer simulations).

**Historical synthetic bundles.** Synthetic `h5/2` bundles written by a library before version 0.2.0 (recipe
`{"synthetic": true}`, no `all_sims`, a made-up w_c map that does not count the synthetic cache's groups) are
historical artefacts: the release loaders refuse them (`ALL_SIMS_NOT_TRUE`, and `WCMAP_NOT_FULL_CACHE` for their map)
and the development path loads them. Synthetic bundles written by 0.2.0 declare `all_sims: true` and count the groups.

**What protects what.**

| layer | protects against | does not protect against |
|---|---|---|
| the published manifest digest, pinned after a recorded acceptance | substitution, corruption, an unaccepted bundle presented as accepted | a compromised maintainer, an unprotected registry manifest |
| the loader metadata rules and the separate types | accidental export or load of fold or research models as production; mislabelled recipes using reserved names; a held-out or training simulation list under any key (`simulation_sets/1`) | deliberately forged metadata with a recomputed digest; simulations given by index, digest or prose; at load, a name outside the simulation-name form in a hand-edited block |
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

The production and research rules apply from library version 0.2.0 (`hcd_analysis.__version__`), the first version
that carries them. Every earlier version, 0.1.0 included (for example the release line at `372204f` and the frozen
benchmark's `3496030`), predates them. Differences a user of an earlier version meets:

- synthetic `h5/2` bundles written by an earlier version (recipe without `all_sims`, a made-up w_c map that does not
  count the cache's groups) are refused by the release loaders (`ALL_SIMS_NOT_TRUE`, and `WCMAP_NOT_FULL_CACHE` for
  the map) and load through the development path; synthetic bundles written by 0.2.0 declare `all_sims: true` and
  their made-up w_c map counts the synthetic cache's groups, so their digests differ from earlier ones;
- a bundle whose w_c map does not count every group of its training cache is refused (`WCMAP_NOT_FULL_CACHE`);
- `ReleaseBundle` has 11 fields (`kind` mandatory, `release_verified` defaulting to False); `MemberMetas(...)` needs
  `kind=`; a slice of the metas is a plain list; `ResearchBundle` is a separate type;
- `predictor.Bundle` takes only a release-verified `ReleaseBundle` with its release-verified metas;
- `metas.anchored and metas.lineage_verified` is no longer a production test (both are true after a research load):
  use `metas.release_verified` (or `kind == "release"`);
- `ineligible_reasons` strings are `"[CODE] text"`;
- a member meta that names a proper subset of the cache's simulations anywhere is refused (`SIM_SET:<path>`), and a
  research member's excluded set is derived from every such list (`simulation_sets/1`);
  `eligibility.production_recipe_reasons` takes the keywords `sims` / `n_sims`, `research_member_reasons` the keyword
  `sims`, and a research block may record `RES_*` admission reasons;
- `provenance` must be a JSON object on every write and every load, the development path included;
  `write_manifest_historical` refuses a research marking;
- a digest given to `load_ensemble` / `load_bundle` that is not 64 lowercase hex digits (and not None) is refused
  before the manifest is read;
- `scripts/accept_retrained_ensemble.py` requires `--wc-map` and `--wc-map-sha256`, writes report format
  `hcd_priya.ensemble_acceptance/2`, and its `trained_on_every_simulation` takes a member meta (a recipe raises
  `TypeError`);
- `scripts/export_safe_bundle.py` requires the recipe to state `n_train` and `n_val` and the content checks to PASS.

## The multi-fidelity record (`hcd_priya.mf_record/1`)

**What it is.** The LF members predict the per-class PRIYA-filtered P1D of the low-resolution simulations. The
multi-fidelity (MF) correction turns that into an HR-calibrated prediction, P_MF = P_LF exp(g(z, tau0, k)). g is
measured mode by mode on matched HR/LF pairs, g = log P_HR - log P_LF,backbone, and fitted as a table model that does
not depend on the cosmological parameters (`mf_modes.fit_mode_mf`: log_rho + gbar_z + gbar_tau + one rank-1 z x tau0
term). Because the targets use the LF backbone's predictions, a record is bound to exactly one backbone bundle.

**Layout.** A record is a directory of its own (it is not listed in a bundle's manifest, so a backbone's manifest
digest does not change when an MF is added). It holds exactly:

- `mf_tables.h5`: the root attribute `format = "hcd_priya.mf/1"` and the group `/mf`, flat, holding one contiguous,
  unfiltered, attribute-free, little-endian float64 dataset per table: `log_rho` (K), `gbar_z_tab` (Nz, 4, K),
  `gtau_tab` (Nr, 4, K), `a_k` (4, K), `u_z` (Nz), `u_tau` (Nr), `z_tab` (Nz, ascending), `tau_tab` (the rung index
  0..Nr-1), `tau_by_z` (Nz, Nr, ascending in rung) and `k_com_hmpc` (K, the box modes 2 pi n / L). Nothing else.
- `mf_record.json` (UTF-8 JSON, closed key set; `hcd_analysis/emulator/mf_release.py`):
  - `format` (`"hcd_priya.mf_record/1"`), `mf_version`, `kind` (`release`, `fold` or `historical`), `model_form`;
  - `tables`: `file`, `format`, `sha256`, `bytes`, `n_modes`, `n_z`, `n_rung`;
  - `backbone`: `manifest_sha256` (the backbone bundle's manifest), `kind` (`release` or `research`),
    `cache_sha256` (the backbone's training cache) and `held_out_sims` (a research backbone's held-out simulations);
  - `caches`: `lf` and `hr`, each `{file, sha256, provenance}` (file names only, never paths). The two digests do
    not establish the same thing, and `provenance` says so with a fixed statement from the code
    (`mf_release.CACHE_PROVENANCE`, checked at load): the LF cache is lineage-verified through the bound backbone (it is
    the backbone's training cache), while the HR cache is digest-anchored only and never lineage-verified (it is read
    without a lineage check, and only its per-class P1D rows and row keys are used);
  - `composition`: `id`, `excluded_sims` and `ruling_ids` (the rulings that set it; at least one);
  - `fold`: null, or `{fold_id, holdout_sims, backbone_excludes_holdout}` for a held-out fit;
  - `train_sims`, `n_pairs_train`;
  - `hr_inputs`: the HR cache fields the MF reads (the per-class P1D rows and their row keys) and the
    catalogue-derived HR columns it excludes, with their status;
  - `mean_flux_convention`: the normalisation of the per-class P1D targets as the code computes it (the function
    and line are named), status `SETTLED`: the global convention, in which the four class spectra share the whole
    sample's mean flux, is the production convention; normalising each class by its own mean flux is a diagnostic;
  - `fit_code`: `{commit, dirty}` of the fit;
  - `mode_mapping`: the HR/LF mode-mapping check the builder ran over every matched pair (`tol` = 1e-12, the canonical
    and stored-grid maxima, and `exception`). The canonical mapping holds to 1e-12 on every pair; so do the stored
    grids, except, when `exception` is not null, the pairs of one recorded design point (`mf_release.D1_EXCEPTION`:
    its LF and HR run names, whose parameter values differ in their last quoted digits) below the recorded bound
    5e-10, with their count and largest error. Any other exception, bound or tolerance is refused;
  - `validated_z_min`: the lower edge of the frozen grouped-validation domain (z 2.2-4.6; the held-out validation
    is reported separately); at build time this is the domain edge, not a passed validation. It is a float set by
    the builder (`--validated-z-min`, 2.2 by default, `mf_release.VALIDATED_Z_MIN`); it may be raised, never lowered.
    The predictor refuses an LF+MF prediction below the larger of it and the tables' lowest redshift.

A published record comes with the sha256 of its `mf_record.json`; the record pins the tables by sha256 and size.

**Loading.** `mf_release.load_mf_record(dir, expected_record_sha256=..., backbone_manifest_sha256=..., kind="release",
composition=None, lf_cache_sha256=None, hr_cache_sha256=None)` reads the record through one no-follow descriptor,
compares its sha256 with the published one before parsing, validates the closed schema and the kind rules, compares the
tables' sha256 before libhdf5 parses them, and walks the tables file with the same strict rules as a member (no links,
no attributes but the root `format`, no chunked, filtered, compact, external or virtual storage, finite values only).
It refuses: another backbone, another kind than the one asked for (a `historical` record loads only when asked for by
kind; it is never applied as the release MF), a composition or cache digest other than the expected ones, any extra
file, a record whose `hr_inputs` name a catalogue-derived HR column or give those columns another status, a
`mean_flux_convention` other than the code's, and a legacy `.npz` product.

**Kind rules.** Every record's LF cache is its backbone's training cache (`caches.lf.sha256` equals
`backbone.cache_sha256`): the targets were formed with that backbone. A `release` record has a release backbone, no
fold and a clean fit tree (`fit_code.dirty` false). A `fold` record names its held-out group, which lies outside its
training and excluded sets; `backbone_excludes_holdout` must equal "the group is among the backbone's held-out
simulations", so a record built with a production backbone (in-sample LF for the held-out group) says so.

**In the predictor.** `predictor.load_bundle(bundle_dir, sha256, mf_dir=..., mf_record_sha256=...)` applies a release
record to a production bundle; `predictor.load_research_bundle(..., mf_dir=..., mf_record_sha256=...)` applies a fold
record to its research backbone (genuine held-out predictions for grouped cross-validation when that backbone excludes
the held-out group). Both refuse a record whose backbone manifest, training cache, backbone kind, held-out simulations
or box modes are not the bundle's. The predictor applies only what it re-derives from the record's own anchored
bytes (`mf_release.reverify`; a loaded `MFRecord` keeps them): a record built or changed in memory (another record's
tables, a relabelled kind) is refused, so `provenance["mf"]` always describes the tables applied.
`predict(..., fidelity="LF")` gives the LF prediction of the same bundle, bit for bit what the bundle loaded without
an MF gives. Below the larger of the lowest redshift of the record's tables and the record's `validated_z_min` the
LF+MF prediction is refused (`MFRangeError`), never clamped: the correction was not measured there or it lies outside the frozen grouped-validation domain,
so only `fidelity="LF"` is available. Above the highest redshift, and outside the tau0 tables, g is held at its edge value and the prediction is
flagged (`mf_z_clamped`, `mf_tau0_clamped`). Above z 4.6 (`mf_release.VALIDATED_Z_MAX`, the upper edge of the frozen grouped-validation domain (z 2.2-4.6; the held-out validation is reported separately);
not a record field) an LF+MF prediction keeps its values but is flagged `mf_z_above_validated`, also
inside the tables, because it lies outside that domain; an LF prediction is never so flagged.

**Writing.** `mf_release.write_mf_record(out_dir, tables, ...)` validates everything first, then creates `out_dir`
(an existing directory, file or symlink is refused: a record is written once), writes the tables through the
temporary file's own descriptor and publishes them by hard link, then writes the record last.
`scripts/build_mf_release.py` builds a record from digest-pinned caches and an anchored backbone; it refuses an existing
output directory and a release build from a dirty tree before reading any input.

**Historical MF products.** The gate D product (`mf_modes_all6.npz`, `mf_modes.save_mode_mf`) is the historical `.npz`
container, read with `allow_pickle=False` by `products.load_mf_product`, which checks the LF ensemble, the LF and HR
cache digests and the composition (`"all"` for a product that records none). It is not a release record and the
release loader refuses it.

## The incidence MF record (`hcd_priya.incidence_mf_record/1`)

The P1D record above corrects the class spectra only. The incidence dN/dX of the three HCD classes, and the
column-density distribution f(N_HI) (the CDDF), also differ between the low- and high-resolution simulations. Their
correction is a separate, small record (`hcd_analysis/emulator/incidence_mf.py`):

- **What it holds.** A fixed table `R_c(z)`: at each redshift, the mean over the matched HR simulations of
  `(dN/dX)_HR / (dN/dX)_LF`, per class (LLS, subDLA, DLA). And the same kind of table per column-density bin,
  `R(z, N_HI)`, for the CDDF, with the bin centres. The tables do not depend on cosmology or on the mean flux.
- **How it is used.** `dndx_MF = dndx_LF * R_c(z)`: linear in z between the tabulated redshifts, held constant beyond
  them. The predictor applies it to an LF+MF prediction before the map from dN/dX to sightline fractions, so `w_c`
  follows; the class spectra are not touched. The predictor returns no CDDF, so the CDDF table is stored and not
  applied.
- **Where the numbers come from.** The tables are built and evaluated by the functions that have been in
  `hcd_analysis/emulator/multifidelity.py` since June 2026 (`build_dndx_res_corr`, `build_cddf_res_corr`,
  `dndx_res_factor`, `apply_dndx_res_corr`, `cddf_res_factor`); the record is their container.
  `scripts/build_incidence_mf.py` builds one from two digest-pinned inputs: the LF training cache, and an HR
  incidence product derived from the rebuilt HR absorber catalogues. The HR cache is not read: its catalogue-derived
  columns stay refused, as stated above.
- **Layout.** One directory with one file, `incidence_mf_record.json`: strict UTF-8 JSON, a closed key set, finite
  numbers, the tables included as lists. The record is anchored by the sha256 of that file.

```text
{"format": "hcd_priya.incidence_mf_record/1", "version": "<text>", "model_form": "<the code's statement>",
 "class_order": ["LLS", "subDLA", "DLA"],
 "tables": {"z": [Nz], "n_sims": [Nz], "dndx_ratio": [[3] x Nz],
            "log_nhi_centres": [Nbin], "cddf_ratio": [[Nbin] x Nz]},
 "lf_cache": {"file": "<name>", "sha256": "<64 hex>", "source": "<the code's statement>"},
 "hr_incidence": {"file": "<name>", "sha256": "<64 hex>", "source": "<the code's statement>",
                  "groups_not_in_product": ["<an HR snapshot group the product does not hold>", ...]},
 "composition": {"id": "<text>", "excluded_sims": [...], "ruling_ids": [...]},
 "train_sims": [...], "n_pairs": N, "fit_code": {"commit": "<40 hex>", "dirty": false}}
```

- `n_sims` is the number of matched pairs behind each redshift's dN/dX mean, and `n_pairs` their sum. The fit refuses
  a pair whose dN/dX is not finite and positive, so every pair counts in every class and no cell is filled in. A
  redshift with one pair is a ratio of one simulation, not a mean. A CDDF cell is the mean over the pairs with both
  values positive, and 1 where there is none.
- `lf_cache.source` and `hr_incidence.source` are the code's statements of what it establishes about each input,
  and it is little: both files are pinned by their sha256. The code verifies no lineage of the HR incidence product;
  that it was derived from the rebuilt HR catalogues is the statement of whoever built the record.
  `groups_not_in_product` is recorded as given, and is not compared with the product.
- `composition` says which HR simulations were left out and under which ruling; `train_sims` are those used. The
  builder has no default composition.
- The loader (`incidence_mf.load_record`) compares the file's sha256 with the published one before parsing, refuses
  any other file in the directory, an unknown key, a statement that differs from the code's, a table that is not
  finite, positive and consistent in shape, and a record fitted from a tree with uncommitted changes.
- The predictor applies a record only when its `lf_cache.sha256` equals the training cache of the backbone, as
  re-derived from the member metas, and only together with the P1D MF record.
- **Redshifts outside the table.** At a snapshot redshift the table does not hold, the factor is interpolated
  between the neighbouring rows and the prediction is flagged `incidence_mf_z_interpolated`; beyond the table it is
  held at the edge row and flagged `incidence_mf_z_clamped`. This differs from the P1D record, which is refused
  below its tables.
- **Not validated here:** the w_c map was fitted on LF counts. That it still holds when dN/dX is scaled to the HR
  incidence has not been tested.

## The release manifest (`hcd_priya.release/1` and `hcd_priya.release/2`)

An LF+MF release is published under ONE digest, the sha256 of a small `release.json` in the release directory
(`hcd_analysis/emulator/release_manifest.py`):

```json
{"format": "hcd_priya.release/1", "release_version": "<text>",
 "backbone": {"dir": "<name>", "manifest_sha256": "<64 hex>"},
 "mf": {"dir": "<name>", "record_sha256": "<64 hex>"}}
```

`backbone.dir` holds the `hcd_priya.h5/2` production bundle and `mf.dir` the `hcd_priya.mf_record/1` record; each
`dir` is one plain directory name inside the release directory (never a path, `..` or a symlink), and the two differ.
The key set is closed and the `mf` part is required: an LF-only bundle is not a release of this format and loads with
`load_bundle` as before. Only `release.json` is anchored by the release digest: the loader reads the two parts it
names and ignores every other file in the release directory, which is not part of the release.
`predictor.load_release(release_dir, "<published release sha256>")` reads `release.json`
through a no-follow descriptor, compares its sha256 before parsing, validates it, and loads both parts through
`load_bundle(..., mf_dir=..., mf_record_sha256=...)` with the digests it names; since the record pins its backbone's
manifest, a manifest naming a mismatched pair is refused. The returned bundle's `provenance["release"]` gives the
format, the release version and the release sha256. `release_manifest.write_release_manifest` writes the file once,
into a directory that already holds both parts. The two-digest call (`load_bundle` with `mf_dir` and
`mf_record_sha256`) keeps working.

**Format 2** is format 1 plus at least one optional part from a closed list. Today the list has one entry, the
incidence MF record:

```json
{"format": "hcd_priya.release/2", "release_version": "<text>",
 "backbone": {"dir": "<name>", "manifest_sha256": "<64 hex>"},
 "mf": {"dir": "<name>", "record_sha256": "<64 hex>"},
 "incidence": {"dir": "<name>", "record_sha256": "<64 hex>"}}
```

`load_release` loads the incidence record against the digest the manifest gives and binds it to the backbone's
training cache; an LF+MF prediction of such a release has its `dndx` multiplied by `R_c(z)` (flag
`incidence_mf_applied`, `provenance["incidence_mf"]`). Only `load_release` applies it: the two-digest call
(`load_bundle` with `mf_dir`) on the parts of a format-2 release does not read the incidence record, and returns
the LF `dndx` and `w_c` with no incidence flag. The two formats behave differently on purpose, and the
difference is visible:

| | format 1 | format 2 with an incidence record |
|---|---|---|
| `P_class` under LF+MF | LF times exp(g) | the same |
| `dndx` under LF+MF | the LF model's | the LF model's times `R_c(z)` |
| `w_c` under LF+MF | the map of the LF `dndx` | the map of the corrected `dndx` |
| anything under `fidelity="LF"` | the LF model's | the same |
| flag `incidence_mf_applied` | absent | present |

A release without an optional part is written in format 1, so a release has one representation. A manifest of
format 1 that names an optional part, a manifest of format 2 that names none, and a part outside the list are refused.

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
