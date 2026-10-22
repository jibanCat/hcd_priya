# Release format for emulator members (`hcd_priya.safe/1`)

Status: prototype for the release. The training checkpoints (`.eqx` + `.norm.pkl` + `.meta.json`, written by
`train.save_checkpoint`) remain the working format; this format is what a release would ship.

## Why

`.norm.pkl` is a pickle: loading it can run arbitrary code. `.eqx` (`equinox.tree_serialise_leaves`) is not a pickle,
but its arrays are unnamed and only make sense when they are read into the exact skeleton they were written from.
A release needs files that load without executing anything from the file and that a loader can check name by name.

## Layout

One member `<prefix>`:

| File | Content |
|---|---|
| `<prefix>.safetensors` | every array: network leaves `model<tree path>` (for example `model.enc.layers[0].weight`) and normalizer arrays `norm/<key>/...`, with an integer dict key written `#<int>` |
| `<prefix>.json` | `format`, the schema-2.0 checkpoint `meta`, the tensor index (name, shape, dtype) and the sha256 of the `.safetensors` file |

A bundle directory adds `manifest.json`: the format, the member order, and every file's sha256 and size.
Tables (multi-fidelity corrections, grids) are HDF5 files with numeric datasets and scalar or string attributes only.

The `.safetensors` file follows the public safetensors layout: an 8-byte little-endian header length, a JSON header that
gives each tensor's dtype code (`F64`, `F32`, `I64`, `I32`, `U8`, `BOOL`), shape and byte range, then the raw
little-endian buffer. `hcd_analysis/emulator/safe_io.py` reads and writes it with numpy alone; the official
`safetensors` package reads the same files (an optional interop test under `tests/external/`).

## Loading contract (`safe_io.load_member`)

1. The JSON sidecar must name this format and carry a schema-2.0 meta that passes the checks of
   `train.load_checkpoint`; `arch_cfg` must have exactly the `Emulator` constructor keys, with positive integer values.
2. The `.safetensors` sha256 must match the sidecar; the header must be strict (no duplicate keys, no gaps, overlaps
   or trailing bytes, offsets consistent with shape and dtype).
3. The network is rebuilt from the code in this repository (`model.Emulator(**arch_cfg)`); nothing named in a file is
   imported or called. Every leaf is then matched by name, shape and dtype; a missing, extra or mismatched tensor
   refuses the load. float64 stays float64 (a float64 member needs `jax_enable_x64`).
4. No pickle is read or written anywhere on this path, including inside metadata or HDF5 attributes.

Export (`safe_io.export_member`, `scripts/export_safe_bundle.py`) never overwrites an existing file. Converting a
legacy checkpoint reads its `.norm.pkl` through `train.load_checkpoint`, so it is done only for this project's own
checkpoints after their digests are verified (`prod_ensemble.verify_manifest`), into a new directory.
