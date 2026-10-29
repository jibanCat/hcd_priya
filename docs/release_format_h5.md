# Release format for emulator members: HDF5 + JSON (`hcd_priya.h5/1`)

Status: the format for released emulator members, adopted subject to an independent review of its loader and a
compatibility test with an older HDF5 library. The trained models themselves are not distributed yet. The training
checkpoints (`.eqx` + `.norm.pkl` + `.meta.json`, written by `train.save_checkpoint`) remain the working format; a
release converts them once (`scripts/export_safe_bundle.py --container h5`). The earlier safetensors-layout prototype
(`docs/release_format.md`) is kept as evidence only and is not a second release format.

## Why

`.norm.pkl` is a pickle: loading it can run arbitrary code. A release needs files that load without executing anything
from the file, whose arrays carry names a loader can check one by one, and that standard tools can open. HDF5 is
already a dependency of this package and is readable everywhere (`h5ls`, `h5dump`, HDFView, any language); the JSON
sidecar is plain text.

## Layout

One member `<prefix>`:

- `<prefix>.h5`: the root attribute `format = "hcd_priya.h5/1"` and nothing else at the root; the group `/model` holds
  one dataset per network leaf, named by its JAX tree path (for example `.enc.layers[0].weight`); the group `/norm`
  holds the output normalizer as nested groups (an integer dictionary key is written `#<int>`). Every dataset is
  contiguous, allocated, unfiltered, attribute-free, little-endian float64.
- `<prefix>.json`: the checkpoint meta (schema 2.0: architecture, seed, k grid, cache digest), the tensor index
  (name, shape, dtype of every dataset) and the sha256 of the `.h5` file.

A bundle is a directory holding the members and `manifest.json`, which lists every file with its sha256 and holds
nothing else. A published bundle comes with the sha256 of its `manifest.json`.

### Keys and names

- **Sidecar** `<prefix>.json` (UTF-8 JSON, finite numbers only, no duplicate keys): `format` (`"hcd_priya.h5/1"`),
  `meta` (the schema-2.0 checkpoint meta), `tensors` (`{name: {"shape": [...], "dtype": "F64"}}`), `h5_sha256` (hex
  sha256 of the `.h5` bytes) and `h5_file` (the `.h5` file name, `<prefix basename>.h5`).
- **Manifest** `manifest.json`: `format`, `members` (member names in order), `files` (`[{"path", "sha256", "bytes"}]`,
  exactly the members' `.h5` and `.json`) and `provenance` (free-form).
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

`load_ensemble` is the release entry point: it loads the bundle and refuses members that do not share one
architecture, k grid and normalizer. `load_bundle` returns the members without that check. `load_member(prefix)` loads
one member, anchored only by its own sidecar unless the manifest's digests are passed
(`expected_sha256`, `expected_json_sha256`); a release therefore loads through `load_ensemble`.
`verify_manifest(bundle_dir)` checks a bundle without loading it. The `expected_manifest_sha256` keyword is required:
pass the published digest (passing `None` loads an unanchored development bundle and says so in the call).

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
through the same descriptor that wrote the file, never by reopening its name. `write_manifest` writes the bundle
manifest.

## Inspecting a member with standard tools

```
h5ls -r prod_member.h5          # every dataset and its shape
python -c "import json; print(json.load(open('prod_member.json'))['meta']['arch_cfg'])"
```
