# Prototype pickle-free format for emulator members (`hcd_priya.safe/1`)

Status: PROTOTYPE, kept as evidence; it is NOT the release format. The release format is HDF5 + JSON
(`hcd_priya.h5/1`), developed separately and adopted subject to its loader hardening, an independent review and a
compatibility test with an older HDF5 library. The training checkpoints (`.eqx` + `.norm.pkl` + `.meta.json`, written by
`train.save_checkpoint`) remain the working format. Nothing here is a second required release format.

## Why

`.norm.pkl` is a pickle: loading it can run arbitrary code. `.eqx` (`equinox.tree_serialise_leaves`) is not a pickle,
but its arrays are unnamed and only make sense when they are read into the exact skeleton they were written from.
A release needs files that load without executing anything from the file and that a loader can check name by name.

## Layout

One member `<prefix>`:

| File | Content |
|---|---|
| `<prefix>.safetensors` | every array: network leaves `model<tree path>` (for example `model.enc.layers[0].weight`) and normalizer arrays `norm/<key>/...`, with an integer dict key written `#<int>` (canonical decimal) |
| `<prefix>.json` | `format`, the schema-2.0 checkpoint `meta`, the tensor index (name, shape, dtype), `safetensors_file` (`<prefix basename>.safetensors`) and its sha256 |

A bundle is a directory holding the members and `manifest.json` (the format, the member names in order, and every
file's sha256 and size) and nothing else. Tables (multi-fidelity corrections, grids) are HDF5 files with numeric
datasets and string or numeric scalar attributes only.

The `.safetensors` file follows the public safetensors layout: an 8-byte little-endian header length, a JSON header that
gives each tensor's dtype code (`F64`, `F32`, `I64`, `I32`, `U8`, `BOOL`), shape and byte range, then the raw
little-endian buffer. `hcd_analysis/emulator/safe_io.py` reads and writes it with numpy alone. The official Python
`safetensors` package (0.4.5) reads files written here and vice versa for every dtype code above (an optional test under
`tests/external/`); other readers of the layout are not tested.

## Loading contract

`safe_io.load_bundle_historical(bundle_dir, expected_manifest_sha256=..., reason=...)` is the prototype's entry point.
The prototype carries no lineage, so the load is a historical read with a non-empty reason, never a release model (the
digest keyword is required; `None` explicitly loads an unanchored bundle, for development only):

1. The manifest (whose own sha256 the caller supplies) must list exactly the two files of each member under plain
   names (no directories, no symlinks), with matching sizes and sha256, and the directory must hold no other file.
2. Each member is loaded with the manifest's digests (`load_member`), hashing the same bytes it parses. The sidecar must
   name this format, `safetensors_file` must be the member's own file, and the meta must pass the schema-2.0 checks of
   `train.load_checkpoint`; `arch_cfg` must have exactly the `Emulator` constructor keys, with integers in a bounded
   range, and the seed must be a 32-bit unsigned integer.
3. The header is parsed strictly: UTF-8 JSON object, no duplicate keys, known dtype codes, at most 8 dimensions of
   bounded size, offsets consistent with shape and dtype (computed with unbounded integers), and no gap, overlap or
   trailing byte; the whole header is validated before any tensor is copied out of the file.
4. The network's tensor names, shapes and dtypes are derived from the code in this repository
   (`model.Emulator(**arch_cfg)` under `equinox.filter_eval_shape`, so nothing is allocated from file-controlled sizes);
   a missing, extra or mismatched tensor, or any tensor outside `model`/`norm/`, refuses the load. float64 stays
   float64 (a float64 member needs `jax_enable_x64`).
5. The normalizer must have the production structure (`f_nhi`, `dndx`, `P_filt` with `mu_marg`, `sig_marg`, `sig_cosmo`
   and integer-keyed `cell_mean`, `delta`), with shapes set by the network and every array float64.
6. No pickle is read or written anywhere on this path, including inside metadata or HDF5 attributes. The HDF5 reader
   refuses nested or dot names, soft and external links, external or virtual storage, variable-length data and
   non-scalar or non-UTF-8 attributes (including surrogate-escaped strings).

Writes never overwrite: every file (tensors, sidecars, manifest and HDF5 tables) is written to a new temporary name
(created exclusively, never through a symlink, with the mode any new file gets under the umask and the directory's
default ACLs), fsynced, and published with a hard link that fails if the target exists (a symlink at the target is
refused, never followed). HDF5 dataset names must be plain names (letters, digits, `_`, `.`, `-`; not starting with
`.` or `-`), the same rule on write and on read. Converting a legacy checkpoint (`scripts/export_safe_bundle.py`) reads its `.norm.pkl` through
`train.load_checkpoint`, so it is done only for this project's own checkpoints, after the production manifest matches a
digest given on the command line and the members' digests are verified (`prod_ensemble.verify_manifest`), into a new
directory.
