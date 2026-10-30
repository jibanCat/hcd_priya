# Release format for emulator bundles: HDF5 + JSON (`hcd_priya.h5/2`)

Status: the format for released emulator bundles, adopted subject to an independent review of its loader and a
compatibility test with an older HDF5 library. The trained models themselves are not distributed yet. The training
checkpoints (`.eqx` + `.norm.pkl` + `.meta.json`, written by `train.save_checkpoint`) remain the working format; a
release converts them once (`scripts/export_safe_bundle.py`). `hcd_priya.h5/2` adds to the earlier `hcd_priya.h5/1`
a lineage block and the dN/dX-to-w_c map; an `h5/1` bundle loads only through the separately named historical loaders
(below). The earlier safetensors-layout prototype (`docs/release_format.md`) is kept as evidence only and is not a
second release format.

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
  file name, `"wc_map.json"`), `lineage` (the lineage block, below) and `provenance` (free-form).
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
  - `n_members`, `release_eligible` and `ineligible_reasons` (computed when the block is made, never passed in).
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

The public predictor wraps this (`hcd_analysis/predictor.py`): `hp.load_bundle(bundle_dir, "<digest>")` (None refused),
`emu.predict(params, z, tau_eff=...)` (the class spectra, the templates, w_c from the bundled map, the members' spread
as a seed-spread diagnostic, provenance) and `hp.add_hcd(pred, A)` (free amplitudes; only A_c >= 0 and
1 - sum_c A_c >= 0 are imposed).

`load_ensemble` is the release entry point. With the published digest it requires format `h5/2` and a lineage block
that establishes a clean lineage: every finder id in the release allow-list, an eligible training cache (its carried
rows labelled and promoted), the members' commit known and the export's tree clean, the w_c map eligible and fitted to
the block's cache and equal to the listed file, and every member trained on the block's cache and lineage record at the
block's commit from a clean tree; otherwise it refuses the bundle. It also refuses members that do not share one
architecture, k grid, parameter box and normalizer. `load_bundle` returns the members after the same checks;
`load_release` (digest required) also returns the parsed w_c map and the block. `load_member(prefix)` loads one member,
anchored only by its own sidecar unless the manifest's digests are passed (`expected_sha256`, `expected_json_sha256`);
a release therefore loads through `load_ensemble`. `verify_manifest(bundle_dir)` checks a bundle without loading it.
The `expected_manifest_sha256` keyword is required: pass the published digest. Passing `None` loads an unanchored
development bundle: the lineage is not enforced, an `UnanchoredBundleWarning` is issued and the returned metas say so
(`metas.anchored` is False). The block is a declaration pinned by the manifest digest: each product's builder verified
its own inputs when it ran, and the loader checks that the declaration is complete, consistent and eligible.

**Historical bundles.** An `h5/1` bundle (written before bundles carried lineage) is refused by the release loaders;
`load_ensemble_historical(bundle_dir, expected_manifest_sha256=..., reason=...)` and `load_bundle_historical` read it
with every `h5/1` format check and no lineage, log the reason, and mark the metas (`metas.historical_reason`).

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
  than 255 characters, non-finite values, more objects, depth or bytes than the bounds in `safe_h5.py`. Shapes of the network and of the normalizer are checked from metadata before any data is
  read. Any HDF5 error becomes a `SafeFormatError`.
- **Never a silent down-cast.** A process without JAX float64 is refused.

## Writing

`safe_h5.export_member(prefix, model, meta, norm)` writes one member and never overwrites: each file is written through
the descriptor of a new temporary file and published by hard link only if that name still holds it. Files are
byte-identical across exports for a fixed HDF5 library version (no timestamps are stored). The writer limits the HDF5
object versions to those libhdf5 1.10 reads (`libver=("earliest", "v110")`). The sidecar's `h5_sha256` is computed
through the same descriptor that wrote the file, never by reopening its name. `export_wc_map` copies the w_c map in;
`write_manifest(bundle_dir, prefixes, provenance, lineage=block)` writes the manifest. The release export
(`scripts/export_safe_bundle.py`) refuses ineligible inputs before writing anything. Synthetic, self-consistent `h5/2`
bundles for tests and CI: `hcd_analysis.emulator.release_testing.write_synthetic_bundle`.

## Inspecting a member with standard tools

```
h5ls -r prod_member.h5          # every dataset and its shape
python -c "import json; print(json.load(open('prod_member.json'))['meta']['arch_cfg'])"
```
