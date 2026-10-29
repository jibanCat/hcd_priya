"""Pickle-free HDF5 + JSON release format for emulator members: format ``hcd_priya.h5/1`` (contract and usage in
docs/release_format_h5.md).

One member ``<prefix>``:
  ``<prefix>.h5``    root attribute ``format`` only; group ``/model`` holds one dataset per network leaf, named by its
                     JAX tree path (for example ``.enc.layers[0].weight``); group ``/norm`` holds the normalizer as
                     nested groups (an integer dict key is written ``#<int>``). Every dataset is contiguous, allocated,
                     unfiltered, attribute-free, little-endian float64.
  ``<prefix>.json``  the schema-2.0 checkpoint meta, the tensor index (name, shape, dtype) and the sha256 of the .h5.
A bundle directory holds the members and ``manifest.json`` (the strict manifest of ``safe_io``) and nothing else.

Loading: every file is read through one descriptor that does not follow a symlink, must be a regular file and is
size-bounded before it is read. The .h5 bytes are read once, hashed and compared with the sidecar (and with the
manifest digest the caller gives ``load_bundle``) BEFORE libhdf5 sees them, and the same in-memory bytes are then
opened, so bytes that were not anchored are never parsed by the HDF5 library. A full, iterative walk of the metadata
then refuses soft and external links, any object reachable twice, named datatypes, external, virtual, compact,
chunked or filtered storage, storage that is not allocated inside the file, any attribute except a scalar-string root
``format`` (its shape and type are checked before its value is read), a user block, anything outside ``/model`` and
``/norm``, a group in ``/model``, a group without datasets in ``/norm``, a link name longer than 255 characters
(checked before any path is built), and more objects or bytes than the bounds below. Names, shapes and dtypes of the
model AND the normalizer are checked against the code-derived structure before any data is read; values must be
finite. Any other HDF5 error becomes a SafeFormatError. No pickle is read or written."""
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
_MAX_TOTAL_BYTES = 1 << 28                   # 256 MiB of declared dataset bytes per member
_MAX_FILE_BYTES = _MAX_TOTAL_BYTES + (1 << 24)   # the .h5 read into memory: the datasets plus 16 MiB of metadata
_LIBVER = ("earliest", "v110")               # files stay readable by libhdf5 1.10
_MAX_OBJECTS = 4096                          # groups + datasets per member
_MAX_DEPTH = 8                               # group nesting below /model or /norm
_MAX_NAME_CHARS = 255                        # one link name (paths are built from names, so this bounds them too)
SafeFormatError = S.SafeFormatError


def _h5():
    import h5py
    return h5py


# --------------------------------------------------------------------------------------------- export
def export_member(prefix, model, meta, norm):
    """Write ``<prefix>.h5`` and ``<prefix>.json`` for one member. Never touches existing files; the .h5 is written
    through the temporary file's own descriptor and published by hard link only if that name still holds it."""
    h5py = _h5()
    hp, js = prefix + EXT, prefix + ".json"
    for p in (hp, js):
        if os.path.lexists(p):
            raise SafeFormatError(f"{p} exists: export never overwrites")
    S._validate_meta(meta, prefix)                             # includes: plain finite JSON, before anything is written
    tensors = {**S.model_tensors(model), **S.norm_tensors(norm)}
    for n, a in tensors.items():
        if S._code(a.dtype) != "F64":
            raise SafeFormatError(f"tensor {n!r} is {a.dtype}: this container stores float64 only")
    S._require_finite(tensors, prefix)

    def writer(fh):
        with h5py.File(fh, "w", libver=_LIBVER) as f:
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
    tmp, held, h5_sha = S._write_tmp(hp, EXT, writer)               # hashed through the writer's own descriptor
    S._publish(tmp, hp, held)
    index = {n: {"shape": list(np.shape(a)), "dtype": "F64"} for n, a in sorted(tensors.items())}
    doc = {"format": FORMAT, "meta": meta, "tensors": index, "h5_sha256": h5_sha,
           "h5_file": os.path.basename(hp)}
    S._write_exclusive(js, [json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")])
    return hp, js


# --------------------------------------------------------------------------------------------- strict reading
def _check_dataset(d, name, where, raw_len):
    """Storage and dtype of one dataset from its metadata only; returns its byte size."""
    h5py = _h5()
    dcpl = d.id.get_create_plist()
    layout = dcpl.get_layout()
    bad = [w for w, hit in (("virtual", d.is_virtual), ("external", bool(d.external)), ("chunked", d.chunks is not None),
                            ("filtered", dcpl.get_nfilters() != 0 or d.compression is not None or d.shuffle
                             or d.fletcher32 or d.scaleoffset is not None),
                            ("compact", layout == h5py.h5d.COMPACT)) if hit]
    if bad:
        raise SafeFormatError(f"{where}: dataset {name!r} is {' and '.join(bad)}; only contiguous unfiltered storage")
    if layout != h5py.h5d.CONTIGUOUS:
        raise SafeFormatError(f"{where}: dataset {name!r} has storage layout {layout}, only contiguous storage")
    if d.id.get_type().committed():
        raise SafeFormatError(f"{where}: dataset {name!r} uses a named datatype")
    if h5py.check_vlen_dtype(d.dtype) is not None or d.dtype.names is not None or d.dtype != _F64:
        raise SafeFormatError(f"{where}: dataset {name!r} has dtype {d.dtype}, expected little-endian float64")
    nbytes = int(d.size) * _F64.itemsize
    if nbytes > _MAX_DATASET_BYTES:
        raise SafeFormatError(f"{where}: dataset {name!r} exceeds the size bound")
    offset = d.id.get_offset()
    if nbytes and (d.id.get_storage_size() != nbytes or offset is None or offset < 0 or offset + nbytes > raw_len):
        raise SafeFormatError(f"{where}: dataset {name!r} has no allocated storage inside the file")
    return nbytes


def _walk(group, path, where, raw_len, seen):
    """Iterative walk below ``group``: every link a hard link to an attribute-free group or dataset reached exactly
    once (by object address and reference count), within the depth and object bounds. Returns {path: dataset} and the
    total byte size and the group paths, from metadata only."""
    h5py = _h5()
    out, total, groups, stack = {}, 0, [], [(group, path, 0)]
    while stack:
        grp, gpath, depth = stack.pop()
        if len(grp) > _MAX_OBJECTS - len(seen):                   # a link count, before any name is read
            raise SafeFormatError(f"{where}: more than {_MAX_OBJECTS} objects")
        for name in grp.keys():
            if len(name) > _MAX_NAME_CHARS:                       # before any path is built from it
                raise SafeFormatError(f"{where}: a link name below {gpath} is longer than {_MAX_NAME_CHARS} "
                                      f"characters")
            p = f"{gpath}/{name}"
            link = grp.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise SafeFormatError(f"{where}: {p} is a {type(link).__name__}, only hard links are allowed")
            obj = grp[name]
            info = h5py.h5o.get_info(obj.id)
            key = (info.fileno, info.addr)
            if info.rc != 1 or key in seen:
                raise SafeFormatError(f"{where}: {p} is reachable through several links")
            seen.add(key)
            if len(seen) > _MAX_OBJECTS:
                raise SafeFormatError(f"{where}: more than {_MAX_OBJECTS} objects")
            if len(obj.attrs):
                raise SafeFormatError(f"{where}: {p} carries attributes")
            if isinstance(obj, h5py.Group):
                if depth + 1 > _MAX_DEPTH:
                    raise SafeFormatError(f"{where}: {p} is nested deeper than {_MAX_DEPTH}")
                groups.append(p)
                stack.append((obj, p, depth + 1))
            elif isinstance(obj, h5py.Dataset):
                total += _check_dataset(obj, p, where, raw_len)
                out[p] = obj
            else:
                raise SafeFormatError(f"{where}: {p} is a {type(obj).__name__}, not a group or dataset")
    return out, total, groups


def read_member_arrays(raw, where, want_model, norm_check):
    """Strictly parse the anchored .h5 bytes ``raw``. ``want_model`` = [(name, shape, dtype)] of the network derived
    from code; ``norm_check(tensors)`` validates the normalizer structure. Both are checked on metadata BEFORE any data
    is read (the normalizer through zero-memory views of the declared shapes). Returns ``{tensor name: array}``."""
    h5py = _h5()
    try:
        with h5py.File(io.BytesIO(raw), "r") as f:
            if f.userblock_size:
                raise SafeFormatError(f"{where}: a user block is not allowed")
            fmt = None
            if len(f.attrs) == 1 and list(f.attrs.keys()) == ["format"]:   # a count, then one name; no value yet
                aid = f.attrs.get_id("format")
                if aid.shape == () and aid.get_type().get_class() == h5py.h5t.STRING:
                    fmt = f.attrs["format"]                        # one scalar string, at most the file's size
            if not isinstance(fmt, str) or fmt != FORMAT:
                raise SafeFormatError(f"{where}: root attributes must be exactly a scalar string format = {FORMAT}")
            if len(f) != 2 or set(f.keys()) != {"model", "norm"}:
                raise SafeFormatError(f"{where}: top level must hold exactly /model and /norm")
            seen = set()
            for top in ("model", "norm"):
                link = f.get(top, getlink=True)
                if not isinstance(link, h5py.HardLink) or not isinstance(f[top], h5py.Group) or len(f[top].attrs):
                    raise SafeFormatError(f"{where}: /{top} must be a plain hard-linked group")
                info = h5py.h5o.get_info(f[top].id)
                if info.rc != 1 or (info.fileno, info.addr) in seen:
                    raise SafeFormatError(f"{where}: /{top} is reachable through several links")
                seen.add((info.fileno, info.addr))
            found, tm, model_groups = _walk(f["model"], "/model", where, len(raw), seen)
            norm_found, tn, norm_groups = _walk(f["norm"], "/norm", where, len(raw), seen)
            if tm + tn > _MAX_TOTAL_BYTES:
                raise SafeFormatError(f"{where}: the datasets exceed the size bound of a member")
            if model_groups:
                raise SafeFormatError(f"{where}: /model must be flat (it holds the group {model_groups[0]})")
            empty = [g for g in norm_groups if not any(p.startswith(g + "/") for p in norm_found)]
            if empty:
                raise SafeFormatError(f"{where}: the group {empty[0]} holds no dataset")
            model_ds = {"model" + p[len("/model/"):]: d for p, d in found.items()}
            want = {n: (shape, dt) for n, shape, dt in want_model}
            if set(model_ds) != set(want):
                raise SafeFormatError(f"{where}: model datasets differ from the network derived from code: missing "
                                      f"{sorted(set(want) - set(model_ds))[:5]}, extra "
                                      f"{sorted(set(model_ds) - set(want))[:5]}")
            for n, d in model_ds.items():
                shape, dt = want[n]
                if tuple(d.shape) != shape or np.dtype(dt) != _F64:
                    raise SafeFormatError(f"{where}: dataset {n!r} is {d.dtype}{d.shape}, the network expects "
                                          f"{dt}{shape} (a float64 file needs jax_enable_x64; it is never down-cast)")
            norm_ds = {"norm/" + p[len("/norm/"):]: d for p, d in norm_found.items()}
            norm_check({n: np.broadcast_to(np.zeros((), _F64), d.shape) for n, d in norm_ds.items()})
            out = {n: d[...] for n, d in model_ds.items()}
            out.update({n: d[...] for n, d in norm_ds.items()})
    except SafeFormatError:
        raise
    except Exception as e:                                         # anything libhdf5/h5py raises on crafted bytes
        raise SafeFormatError(f"{where}: unreadable HDF5 ({type(e).__name__}: {e})"[:300]) from None
    S._require_finite(out, where)
    return out


def load_member(prefix, *, expected_sha256=None, expected_json_sha256=None):
    """Load one member exported by ``export_member``. Returns ``(model, meta, norm)``; refuses everything
    ``safe_io.load_member`` refuses plus the HDF5-specific structures listed in the module docstring. The .h5 is
    checked against its own sidecar; without ``expected_sha256`` / ``expected_json_sha256`` (the manifest's digests,
    which ``load_bundle`` passes) nothing outside the member anchors the pair, so a release loads through
    ``load_ensemble`` (or ``load_bundle``) with the published manifest digest."""
    import jax
    import jax.numpy as jnp

    js = prefix + ".json"
    raw_js = S.read_regular(js, S.MAX_JSON_BYTES)
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
    raw = S.read_regular(hp, _MAX_FILE_BYTES)
    digest = hashlib.sha256(raw).hexdigest()                  # anchored BEFORE libhdf5 parses anything
    if digest != doc["h5_sha256"] or (expected_sha256 is not None and digest != expected_sha256):
        raise SafeFormatError(f"{hp}: sha256 does not match {js}" + (" or the manifest" if expected_sha256 else ""))
    treedef, want = S._abstract_network(meta["arch_cfg"], meta["seed"])
    shapes = {n: s for n, s, _ in want}
    n_k, n_cddf, n_dndx = (meta["arch_cfg"]["n_k"], shapes["model.head_a.cddf.weight"][0],
                           shapes["model.head_a.dndx.weight"][0])

    def norm_check(views):
        S._validate_norm(S.norm_from_tensors(views), n_k, n_cddf, n_dndx)
    tensors = read_member_arrays(raw, hp, want, norm_check)
    index = doc["tensors"]
    if not isinstance(index, dict) or set(index) != set(tensors):
        raise SafeFormatError(f"{hp}: tensor names differ from the JSON index")
    for n, a in tensors.items():
        e = index[n]
        if not isinstance(e, dict) or e.get("shape") != list(a.shape) or e.get("dtype") != "F64":
            raise SafeFormatError(f"{hp}: tensor {n!r} does not match its JSON index entry")
    norm = S.norm_from_tensors(tensors)
    S._validate_norm(norm, n_k, n_cddf, n_dndx)
    model = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(tensors[n]) for n, _, _ in want])
    return model, meta, norm


# --------------------------------------------------------------------------------------------- bundles
def write_manifest(bundle_dir, member_prefixes, provenance):
    return S.write_manifest(bundle_dir, member_prefixes, provenance, data_ext=EXT, fmt=FORMAT)


def verify_manifest(bundle_dir, *, expected_sha256=None, allow_extra=False):
    return [p for p, _, _ in S._manifest_entries(bundle_dir, expected_sha256, allow_extra, data_ext=EXT, fmt=FORMAT)]


def load_ensemble(bundle_dir, *, expected_manifest_sha256):
    """The release entry point: ``load_bundle``, then require every member to share one network architecture, one k
    grid and one normalizer (a prediction applies member 0's normalizer to every member). Returns
    ``(ensemble.EnsembleEmulator, [meta per member], norm)``."""
    from .ensemble import EnsembleEmulator
    members = load_bundle(bundle_dir, expected_manifest_sha256=expected_manifest_sha256)
    _, meta0, norm0 = members[0]
    t0 = S.norm_tensors(norm0)
    for i, (_, meta, norm) in enumerate(members[1:], 1):
        if meta["arch_cfg"] != meta0["arch_cfg"] or meta["k_com_hmpc"] != meta0["k_com_hmpc"]:
            raise SafeFormatError(f"{bundle_dir}: member {i} has another architecture or k grid than member 0")
        t = S.norm_tensors(norm)
        if set(t) != set(t0) or any(t[n].shape != t0[n].shape or t[n].tobytes() != t0[n].tobytes() for n in t0):
            raise SafeFormatError(f"{bundle_dir}: member {i} does not share member 0's normalizer")
    return EnsembleEmulator([m for m, _, _ in members]), [meta for _, meta, _ in members], norm0


def load_bundle(bundle_dir, *, expected_manifest_sha256):
    """Verify the manifest, then load every member with the manifest's digests (anchor keyword required; ``None``
    explicitly loads an unanchored development bundle)."""
    return [load_member(p, expected_sha256=s, expected_json_sha256=j)
            for p, s, j in S._manifest_entries(bundle_dir, expected_manifest_sha256, data_ext=EXT, fmt=FORMAT)]


sha256_file = S.sha256_file
