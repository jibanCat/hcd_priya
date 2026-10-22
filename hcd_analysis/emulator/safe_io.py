"""Pickle-free release format for the emulator: format ``hcd_priya.safe/1`` (decision record: docs/release_format.md).

Layout of one LF member ``<prefix>``:
  ``<prefix>.safetensors``  every array: the network leaves (``model.<tree path>``) and the normalizer
                            (``norm/<key>/...``; an integer dict key is written ``#<int>``), in the standard
                            safetensors layout (8-byte little-endian header length, a JSON header giving each tensor's
                            dtype, shape and byte range, then the raw little-endian buffer). Written and read here with
                            numpy only; the official ``safetensors`` package reads the same files (tests/external).
  ``<prefix>.json``         the schema-2.0 checkpoint meta, the tensor index (name, shape, dtype) and the sha256 of the
                            .safetensors file.
A bundle directory adds ``manifest.json`` (format, member order, every file's sha256 and size, provenance).
Tables (MF corrections, grids) go to HDF5 with numeric datasets and JSON-able attributes only (``write_table_h5``).

Loading never imports or calls anything named in a file: the network is rebuilt from reviewed code
(``model.Emulator``) and an architecture config checked against an allowlist, then every leaf is matched by name, shape
and dtype. No pickle is read or written on this path."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile

import numpy as np

FORMAT = "hcd_priya.safe/1"
# safetensors dtype codes this format writes; float64 stays float64 (never down-cast)
_DTYPES = {"F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "I64": np.dtype("<i8"), "I32": np.dtype("<i4"),
           "U8": np.dtype("|u1"), "BOOL": np.dtype("|b1")}
_CODES = {v: k for k, v in _DTYPES.items()}
_MAX_HEADER = 100 * 1024 * 1024
ARCH_KEYS = ("in_dim", "n_k", "n_basis", "baseline_n_layers", "baseline_width")   # Emulator(**arch_cfg)


class SafeFormatError(ValueError):
    """A file, header, tensor set or configuration that does not match the format."""


# --------------------------------------------------------------------------------------------- safetensors layer
def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise SafeFormatError(f"duplicate key {k!r} in a JSON header")
        out[k] = v
    return out


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _code(dtype):
    return _CODES.get(dtype.newbyteorder("<") if dtype.byteorder == ">" else dtype)


def write_safetensors(path, tensors, metadata=None):
    """Write ``{name: array}`` to ``path`` in the safetensors layout (atomic replace). Arrays keep their dtype."""
    if not tensors:
        raise SafeFormatError("no tensors to write")
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        if not isinstance(name, str) or not name or name == "__metadata__":
            raise SafeFormatError(f"invalid tensor name {name!r}")
        a = np.asarray(tensors[name])
        code = _code(a.dtype)
        if code is None:
            raise SafeFormatError(f"tensor {name!r}: dtype {a.dtype} is not supported (numeric dtypes only)")
        buf = np.ascontiguousarray(a, dtype=_DTYPES[code]).tobytes()
        header[name] = {"dtype": code, "shape": [int(s) for s in a.shape], "data_offsets": [offset, offset + len(buf)]}
        chunks.append(buf)
        offset += len(buf)
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise SafeFormatError("metadata must map str to str")
        header["__metadata__"] = dict(metadata)
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".safetensors")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            for c in chunks:
                f.write(c)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_safetensors(path):
    """Read a safetensors file strictly. Returns ``(tensors, metadata)``; every array is a writable copy."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise SafeFormatError(f"{path}: truncated (no header length)")
    (n,) = struct.unpack("<Q", raw[:8])
    if n == 0 or n > _MAX_HEADER or 8 + n > len(raw):
        raise SafeFormatError(f"{path}: header length {n} is invalid for a {len(raw)}-byte file")
    try:
        header = json.loads(raw[8:8 + n].decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SafeFormatError(f"{path}: unreadable header ({e})") from None
    if not isinstance(header, dict):
        raise SafeFormatError(f"{path}: header is not a JSON object")
    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise SafeFormatError(f"{path}: __metadata__ must map str to str")
    buf = memoryview(raw)[8 + n:]
    spans, out = [], {}
    for name, e in header.items():
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            raise SafeFormatError(f"{path}: tensor {name!r} has a malformed entry")
        if e["dtype"] not in _DTYPES:
            raise SafeFormatError(f"{path}: tensor {name!r} has unsupported dtype {e['dtype']!r}")
        shape, off = e["shape"], e["data_offsets"]
        if (not isinstance(shape, list) or not all(_is_int(s) and s >= 0 for s in shape)
                or not isinstance(off, list) or len(off) != 2 or not all(_is_int(o) for o in off)):
            raise SafeFormatError(f"{path}: tensor {name!r} has a malformed shape or offsets")
        a, b = off
        dt = _DTYPES[e["dtype"]]
        if not 0 <= a <= b <= len(buf) or b - a != int(np.prod(shape, dtype=np.int64)) * dt.itemsize:
            raise SafeFormatError(f"{path}: tensor {name!r} offsets {off} do not match its shape/dtype or the file")
        spans.append((a, b, name))
        out[name] = np.frombuffer(buf[a:b], dtype=dt).reshape(shape).copy()
    spans.sort()
    pos = 0
    for a, b, name in spans:
        if a != pos:
            raise SafeFormatError(f"{path}: byte buffer has a gap or an overlap at tensor {name!r}")
        pos = b
    if pos != len(buf):
        raise SafeFormatError(f"{path}: {len(buf) - pos} trailing bytes not owned by any tensor")
    return out, meta


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- tree <-> names
def model_tensors(model):
    """``{"model<tree path>": array}`` for every array leaf of an Equinox model, names from the tree paths."""
    import jax
    leaves, _ = jax.tree_util.tree_flatten_with_path(model)
    out = {}
    for p, x in leaves:
        if not hasattr(x, "shape"):
            raise SafeFormatError(f"non-array leaf at {jax.tree_util.keystr(p)}: the format stores arrays only")
        out["model" + jax.tree_util.keystr(p)] = np.asarray(x)
    return out


def _norm_flatten(d, prefix, out):
    if not isinstance(d, dict):
        raise SafeFormatError(f"normalizer node {prefix!r} is not a dict")
    for k, v in d.items():
        if isinstance(k, bool) or not isinstance(k, (str, int)):
            raise SafeFormatError(f"normalizer key {k!r} under {prefix!r} is neither str nor int")
        if isinstance(k, str) and (not k or "/" in k or k.startswith("#")):
            raise SafeFormatError(f"normalizer key {k!r} under {prefix!r} cannot be encoded")
        seg = f"#{k}" if isinstance(k, int) else k
        name = f"{prefix}/{seg}"
        if isinstance(v, dict):
            _norm_flatten(v, name, out)
        else:
            a = np.asarray(v)
            if a.dtype == object:
                raise SafeFormatError(f"normalizer value {name!r} is not numeric")
            out[name] = a


def norm_tensors(norm):
    """Flatten the nested normalizer dict to ``{"norm/<k1>/<k2>...": array}`` (integer keys written ``#<int>``)."""
    out = {}
    _norm_flatten(norm, "norm", out)
    return out


def norm_from_tensors(tensors):
    """Inverse of ``norm_tensors`` (only names under ``norm/``)."""
    root = {}
    for name in sorted(tensors):
        parts = name.split("/")
        if parts[0] != "norm" or len(parts) < 2:
            continue
        node = root
        keys = [int(p[1:]) if p.startswith("#") else p for p in parts[1:]]
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = tensors[name]
    return root


# --------------------------------------------------------------------------------------------- members
def _validate_arch(arch):
    if not isinstance(arch, dict) or set(arch) != set(ARCH_KEYS):
        raise SafeFormatError(f"arch_cfg keys {sorted(arch) if isinstance(arch, dict) else arch!r} are not the "
                              f"allowlist {sorted(ARCH_KEYS)}")
    for k in ARCH_KEYS:
        if k == "n_basis" and arch[k] is None:                   # train._arch_cfg_from_model: no basis head
            continue
        if not _is_int(arch[k]) or arch[k] <= 0:
            raise SafeFormatError(f"arch_cfg[{k!r}] = {arch[k]!r} is not a positive int")


def _validate_meta(meta, where):
    """The checkpoint-schema-2.0 meta checks of ``train.load_checkpoint``, plus the architecture allowlist and seed."""
    from .schema import CHECKPOINT_SCHEMA_VERSION
    from .train import _SCHEMA2_REQUIRED, _velocity_key_paths
    if not isinstance(meta, dict) or meta.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SafeFormatError(f"{where}: checkpoint schema "
                              f"{meta.get('schema_version') if isinstance(meta, dict) else None!r}, expected "
                              f"{CHECKPOINT_SCHEMA_VERSION}")
    missing = [k for k in _SCHEMA2_REQUIRED if meta.get(k) is None]
    if missing:
        raise SafeFormatError(f"{where}: schema-2.0 meta lacks {missing}")
    if not (isinstance(meta["cache_sha256"], str) and len(meta["cache_sha256"]) == 64):
        raise SafeFormatError(f"{where}: cache_sha256 is not a 64-hex digest")
    if _velocity_key_paths(meta):
        raise SafeFormatError(f"{where}: meta carries a velocity grid key {_velocity_key_paths(meta)}")
    _validate_arch(meta.get("arch_cfg"))
    if not _is_int(meta.get("seed")):
        raise SafeFormatError(f"{where}: seed must be an int")


def export_member(prefix, model, meta, norm):
    """Write ``<prefix>.safetensors`` and ``<prefix>.json`` for one member. Never touches existing files."""
    st, js = f"{prefix}.safetensors", f"{prefix}.json"
    for p in (st, js):
        if os.path.exists(p):
            raise SafeFormatError(f"{p} exists: export never overwrites")
    _validate_meta(meta, prefix)
    tensors = {**model_tensors(model), **norm_tensors(norm)}
    write_safetensors(st, tensors, metadata={"format": FORMAT, "schema_version": str(meta.get("schema_version"))})
    index = {n: {"shape": list(a.shape), "dtype": _code(a.dtype)} for n, a in sorted(tensors.items())}
    doc = {"format": FORMAT, "meta": meta, "tensors": index, "safetensors_sha256": sha256_file(st),
           "safetensors_file": os.path.basename(st)}
    with open(js, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return st, js


def load_member(prefix):
    """Load one member exported by ``export_member``. Returns ``(model, meta, norm)``.

    Refuses: another format version or checkpoint schema, an architecture outside the allowlist, a sha256 mismatch,
    missing or extra tensors, any shape or dtype difference from the model rebuilt from code."""
    import jax
    import jax.numpy as jnp

    from .model import Emulator

    with open(f"{prefix}.json") as f:
        doc = json.load(f, object_pairs_hook=_no_duplicate_keys)
    if doc.get("format") != FORMAT:
        raise SafeFormatError(f"{prefix}.json: format {doc.get('format')!r}, this loader reads {FORMAT}")
    meta = doc.get("meta") or {}
    _validate_meta(meta, f"{prefix}.json")
    st = os.path.join(os.path.dirname(os.path.abspath(f"{prefix}.json")), doc.get("safetensors_file", ""))
    if sha256_file(st) != doc.get("safetensors_sha256"):
        raise SafeFormatError(f"{st}: sha256 does not match {prefix}.json")
    tensors, smeta = read_safetensors(st)
    if smeta.get("format") != FORMAT:
        raise SafeFormatError(f"{st}: format {smeta.get('format')!r}, expected {FORMAT}")
    index = doc.get("tensors") or {}
    if set(index) != set(tensors):
        raise SafeFormatError(f"{st}: tensor names differ from the JSON index")
    for n, a in tensors.items():
        if list(a.shape) != index[n]["shape"] or _CODES.get(a.dtype) != index[n]["dtype"]:
            raise SafeFormatError(f"{st}: tensor {n!r} does not match its JSON index entry")
    skeleton = Emulator(**meta["arch_cfg"], key=jax.random.PRNGKey(meta["seed"]))
    paths, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    want = {"model" + jax.tree_util.keystr(p): x for p, x in paths}
    have = {n for n in tensors if n.startswith("model")}
    if set(want) != have:
        raise SafeFormatError(f"{st}: model tensors differ from the rebuilt network: missing "
                              f"{sorted(set(want) - have)[:5]}, extra {sorted(have - set(want))[:5]}")
    leaves = []
    for p, x in paths:
        n = "model" + jax.tree_util.keystr(p)
        a = tensors[n]
        if a.shape != tuple(x.shape) or a.dtype != np.asarray(x).dtype:
            raise SafeFormatError(f"{st}: tensor {n!r} is {a.dtype}{a.shape}, the network expects "
                                  f"{np.asarray(x).dtype}{tuple(x.shape)} (a float64 file needs "
                                  "jax_enable_x64; it is never down-cast)")
        leaves.append(jnp.asarray(a))
    model = jax.tree_util.tree_unflatten(treedef, leaves)
    return model, meta, norm_from_tensors(tensors)


# --------------------------------------------------------------------------------------------- bundle manifest
def write_manifest(bundle_dir, member_prefixes, provenance):
    """``manifest.json`` tying every file of the bundle together (sha256, size), in member order."""
    files = []
    for pre in member_prefixes:
        for ext in (".safetensors", ".json"):
            p = f"{pre}{ext}"
            files.append({"path": os.path.relpath(p, bundle_dir), "sha256": sha256_file(p), "bytes": os.path.getsize(p)})
    man = {"format": FORMAT, "members": [os.path.relpath(p, bundle_dir) for p in member_prefixes], "files": files,
           "provenance": provenance}
    path = os.path.join(bundle_dir, "manifest.json")
    with open(path, "x") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    return path


def verify_manifest(bundle_dir):
    """Check every file listed in ``manifest.json`` (sha256, size). Returns the member prefixes in order."""
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        man = json.load(f, object_pairs_hook=_no_duplicate_keys)
    if man.get("format") != FORMAT:
        raise SafeFormatError(f"manifest format {man.get('format')!r}, expected {FORMAT}")
    for e in man["files"]:
        p = os.path.join(bundle_dir, e["path"])
        if not os.path.isfile(p) or os.path.getsize(p) != e["bytes"] or sha256_file(p) != e["sha256"]:
            raise SafeFormatError(f"bundle file {e['path']} is missing or does not match the manifest")
    return [os.path.join(bundle_dir, m) for m in man["members"]]


# --------------------------------------------------------------------------------------------- HDF5 tables
def write_table_h5(path, arrays, attrs):
    """Numeric datasets plus JSON-able scalar or string attributes; never overwrites."""
    import h5py
    if os.path.exists(path):
        raise SafeFormatError(f"{path} exists: never overwritten")
    with h5py.File(path, "w-") as f:
        f.attrs["format"] = FORMAT
        for k, v in attrs.items():
            if not isinstance(v, (str, int, float, bool)):
                raise SafeFormatError(f"attribute {k!r} must be a str, int, float or bool")
            f.attrs[k] = v
        for k, v in arrays.items():
            a = np.asarray(v)
            if a.dtype == object or a.dtype.kind not in "fiub":
                raise SafeFormatError(f"dataset {k!r}: dtype {a.dtype} is not numeric")
            f.create_dataset(k, data=a)


def read_table_h5(path, required):
    """Read datasets ``required`` = {name: (dtype, ndim)} exactly; refuses object or variable-length data."""
    import h5py
    out, attrs = {}, {}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise SafeFormatError(f"{path}: format {f.attrs.get('format')!r}, expected {FORMAT}")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        for name, (dt, ndim) in required.items():
            if name not in f:
                raise SafeFormatError(f"{path}: dataset {name!r} missing")
            d = f[name]
            if d.dtype.kind not in "fiub" or d.dtype != np.dtype(dt) or d.ndim != ndim:
                raise SafeFormatError(f"{path}: dataset {name!r} is {d.dtype} ndim {d.ndim}, expected {dt} ndim {ndim}")
            out[name] = d[...]
    return out, attrs
