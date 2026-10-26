"""Pickle-free HDF5 + JSON release format for emulator members: format ``hcd_priya.h5/1`` (prototype; contract in
docs/release_format.md, alternative to the safetensors container of ``safe_io``).

One member ``<prefix>``:
  ``<prefix>.h5``    root attribute ``format`` only; group ``/model`` holds one dataset per network leaf, named by its
                     JAX tree path (for example ``.enc.layers[0].weight``); group ``/norm`` holds the normalizer as
                     nested groups (an integer dict key is written ``#<int>``). Every dataset is contiguous, unfiltered,
                     attribute-free, little-endian float64.
  ``<prefix>.json``  the schema-2.0 checkpoint meta, the tensor index (name, shape, dtype) and the sha256 of the .h5.
A bundle directory holds the members and ``manifest.json`` (the strict manifest of ``safe_io``) and nothing else.

Loading: the .h5 bytes are read once, hashed and compared with the sidecar (and the manifest) BEFORE libhdf5 sees them,
and the same in-memory bytes are then opened, so bytes that were not anchored are never parsed by the HDF5 library.
A full walk refuses soft and external links, hard-link aliases, named datatypes, external or virtual storage,
chunked or filtered datasets, any attribute except the root ``format``, a user block and anything outside ``/model``
and ``/norm``; names, shapes and dtypes are checked against the network derived from code (``safe_io``'s abstract
evaluation) and against a size bound before any data is read. No pickle is read or written."""
from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np

from . import safe_io as S

FORMAT = "hcd_priya.h5/1"
EXT = ".h5"
_F64 = np.dtype("<f8")
_MAX_DATASET_BYTES = 1 << 28                 # 256 MiB per dataset (the production members are a few MB in all)
SafeFormatError = S.SafeFormatError


def _h5():
    import h5py
    return h5py


# --------------------------------------------------------------------------------------------- export
def export_member(prefix, model, meta, norm):
    """Write ``<prefix>.h5`` and ``<prefix>.json`` for one member. Never touches existing files."""
    h5py = _h5()
    hp, js = prefix + EXT, prefix + ".json"
    for p in (hp, js):
        if os.path.lexists(p):
            raise SafeFormatError(f"{p} exists: export never overwrites")
    S._validate_meta(meta, prefix)
    tensors = {**S.model_tensors(model), **S.norm_tensors(norm)}
    for n, a in tensors.items():
        if S._code(a.dtype) != "F64":
            raise SafeFormatError(f"tensor {n!r} is {a.dtype}: this container stores float64 only")
    fd, tmp = S._new_tmp(hp, EXT)
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT
            gm, gn = f.create_group("model"), f.create_group("norm")
            for name in sorted(tensors):
                arr = np.ascontiguousarray(tensors[name], dtype=_F64)
                if name.startswith("model"):
                    gm.create_dataset(name[len("model"):], data=arr)
                else:
                    node = gn
                    parts = name.split("/")[1:]
                    for seg in parts[:-1]:
                        node = node.require_group(seg)
                    node.create_dataset(parts[-1], data=arr)
        S._finalize_tmp(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    S._publish(tmp, hp)
    index = {n: {"shape": list(np.shape(a)), "dtype": "F64"} for n, a in sorted(tensors.items())}
    doc = {"format": FORMAT, "meta": meta, "tensors": index, "h5_sha256": S.sha256_file(hp),
           "h5_file": os.path.basename(hp)}
    S._write_exclusive(js, [json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")])
    return hp, js


# --------------------------------------------------------------------------------------------- strict reading
def _check_dataset(d, name, where):
    h5py = _h5()
    dcpl = d.id.get_create_plist()
    bad = [w for w, hit in (("virtual", d.is_virtual), ("external", bool(d.external)), ("chunked", d.chunks is not None),
                            ("filtered", dcpl.get_nfilters() != 0 or d.compression is not None or d.shuffle
                             or d.fletcher32 or d.scaleoffset is not None)) if hit]
    if bad:
        raise SafeFormatError(f"{where}: dataset {name!r} is {' and '.join(bad)}; only contiguous unfiltered storage")
    if h5py.check_vlen_dtype(d.dtype) is not None or d.dtype.names is not None or d.dtype != _F64:
        raise SafeFormatError(f"{where}: dataset {name!r} has dtype {d.dtype}, expected little-endian float64")
    if d.size * d.dtype.itemsize > _MAX_DATASET_BYTES:
        raise SafeFormatError(f"{where}: dataset {name!r} exceeds the size bound")


def _walk(group, path, out, where):
    """Every link a hard link to a singly-linked, attribute-free group or dataset; datasets collected by path."""
    h5py = _h5()
    for name in group.keys():
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise SafeFormatError(f"{where}: {path}/{name} is a {type(link).__name__}, only hard links are allowed")
        obj = group[name]
        if h5py.h5o.get_info(obj.id).rc != 1:
            raise SafeFormatError(f"{where}: {path}/{name} is reachable through several links")
        if len(obj.attrs):
            raise SafeFormatError(f"{where}: {path}/{name} carries attributes")
        if isinstance(obj, h5py.Group):
            _walk(obj, f"{path}/{name}", out, where)
        elif isinstance(obj, h5py.Dataset):
            _check_dataset(obj, f"{path}/{name}", where)
            out[f"{path}/{name}"] = obj
        else:
            raise SafeFormatError(f"{where}: {path}/{name} is a {type(obj).__name__}, not a group or dataset")


def read_member_arrays(raw, where, want_model):
    """Strictly parse the anchored .h5 bytes ``raw``. ``want_model`` = [(name, shape, dtype)] from the network derived
    from code; model shapes and dtypes are compared BEFORE any data is read. Returns ``{tensor name: array}``."""
    h5py = _h5()
    with h5py.File(io.BytesIO(raw), "r") as f:
        if f.userblock_size:
            raise SafeFormatError(f"{where}: a user block is not allowed")
        if set(f.attrs) != {"format"} or f.attrs["format"] != FORMAT:
            raise SafeFormatError(f"{where}: root attributes must be exactly format = {FORMAT}")
        if set(f.keys()) != {"model", "norm"}:
            raise SafeFormatError(f"{where}: top level must hold exactly /model and /norm")
        for top in ("model", "norm"):
            link = f.get(top, getlink=True)
            if not isinstance(link, h5py.HardLink) or not isinstance(f[top], h5py.Group) or len(f[top].attrs):
                raise SafeFormatError(f"{where}: /{top} must be a plain hard-linked group")
            if h5py.h5o.get_info(f[top].id).rc != 1:
                raise SafeFormatError(f"{where}: /{top} is reachable through several links")
        found = {}
        _walk(f["model"], "/model", found, where)
        model_ds = {"model" + p[len("/model/"):]: d for p, d in found.items()}
        if any("/" in n[len("model"):] for n in model_ds):
            raise SafeFormatError(f"{where}: /model must be flat")
        norm_found = {}
        _walk(f["norm"], "/norm", norm_found, where)
        want = {n: (shape, dt) for n, shape, dt in want_model}
        if set(model_ds) != set(want):
            raise SafeFormatError(f"{where}: model datasets differ from the network derived from code: missing "
                                  f"{sorted(set(want) - set(model_ds))[:5]}, extra {sorted(set(model_ds) - set(want))[:5]}")
        for n, d in model_ds.items():
            shape, dt = want[n]
            if tuple(d.shape) != shape or np.dtype(dt) != _F64:
                raise SafeFormatError(f"{where}: dataset {n!r} is {d.dtype}{d.shape}, the network expects {dt}{shape} "
                                      "(a float64 file needs jax_enable_x64; it is never down-cast)")
        out = {n: d[...] for n, d in model_ds.items()}
        out.update({"norm/" + p[len("/norm/"):]: d[...] for p, d in norm_found.items()})
    return out


def load_member(prefix, *, expected_sha256=None, expected_json_sha256=None):
    """Load one member exported by ``export_member``. Returns ``(model, meta, norm)``; refuses everything
    ``safe_io.load_member`` refuses plus the HDF5-specific structures listed in the module docstring."""
    import jax
    import jax.numpy as jnp

    js = prefix + ".json"
    with open(js, "rb") as fh:
        raw_js = fh.read()
    if expected_json_sha256 is not None and hashlib.sha256(raw_js).hexdigest() != expected_json_sha256:
        raise SafeFormatError(f"{js}: sha256 does not match the manifest")
    doc = S._json_loads(raw_js, js)
    need = ("format", "meta", "tensors", "h5_sha256", "h5_file")
    if not isinstance(doc, dict) or any(k not in doc for k in need):
        raise SafeFormatError(f"{js}: sidecar lacks {[k for k in need if not isinstance(doc, dict) or k not in doc]}")
    if doc["format"] != FORMAT:
        raise SafeFormatError(f"{js}: format {doc['format']!r}, this loader reads {FORMAT}")
    meta = doc["meta"]
    S._validate_meta(meta, js)
    if doc["h5_file"] != os.path.basename(prefix) + EXT:
        raise SafeFormatError(f"{js}: h5_file {doc['h5_file']!r} is not {os.path.basename(prefix)}{EXT}")
    hp = prefix + EXT
    if os.path.islink(hp):
        raise SafeFormatError(f"{hp} is a symlink")
    with open(hp, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()                  # anchored BEFORE libhdf5 parses anything
    if digest != doc["h5_sha256"] or (expected_sha256 is not None and digest != expected_sha256):
        raise SafeFormatError(f"{hp}: sha256 does not match {js}" + (" or the manifest" if expected_sha256 else ""))
    treedef, want = S._abstract_network(meta["arch_cfg"], meta["seed"])
    tensors = read_member_arrays(raw, hp, want)
    index = doc["tensors"]
    if not isinstance(index, dict) or set(index) != set(tensors):
        raise SafeFormatError(f"{hp}: tensor names differ from the JSON index")
    for n, a in tensors.items():
        e = index[n]
        if not isinstance(e, dict) or e.get("shape") != list(a.shape) or e.get("dtype") != "F64":
            raise SafeFormatError(f"{hp}: tensor {n!r} does not match its JSON index entry")
    shapes = {n: s for n, s, _ in want}
    norm = S.norm_from_tensors(tensors)
    S._validate_norm(norm, meta["arch_cfg"]["n_k"], shapes["model.head_a.cddf.weight"][0],
                     shapes["model.head_a.dndx.weight"][0])
    model = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(tensors[n]) for n, _, _ in want])
    return model, meta, norm


# --------------------------------------------------------------------------------------------- bundles
def write_manifest(bundle_dir, member_prefixes, provenance):
    return S.write_manifest(bundle_dir, member_prefixes, provenance, data_ext=EXT, fmt=FORMAT)


def verify_manifest(bundle_dir, *, expected_sha256=None, allow_extra=False):
    return [p for p, _, _ in S._manifest_entries(bundle_dir, expected_sha256, allow_extra, data_ext=EXT, fmt=FORMAT)]


def load_bundle(bundle_dir, *, expected_manifest_sha256):
    """Verify the manifest, then load every member with the manifest's digests (anchor keyword required; ``None``
    explicitly loads an unanchored development bundle)."""
    return [load_member(p, expected_sha256=s, expected_json_sha256=j)
            for p, s, j in S._manifest_entries(bundle_dir, expected_manifest_sha256, data_ext=EXT, fmt=FORMAT)]


sha256_file = S.sha256_file
