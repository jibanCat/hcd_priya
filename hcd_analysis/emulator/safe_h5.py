"""Pickle-free HDF5 + JSON release format for emulator bundles: format ``hcd_priya.h5/2`` (contract and usage in
docs/release_format_h5.md). The earlier ``hcd_priya.h5/1`` (bundles without lineage) loads only through the separately
named ``load_ensemble_historical`` / ``load_bundle_historical``, with a reason.

One member ``<prefix>``:
  ``<prefix>.h5``    root attribute ``format`` only; group ``/model`` holds one dataset per network leaf, named by its
                     JAX tree path (for example ``.enc.layers[0].weight``); group ``/norm`` holds the normalizer as
                     nested groups (an integer dict key is written ``#<int>``). Every dataset is contiguous, allocated,
                     unfiltered, attribute-free, little-endian float64.
  ``<prefix>.json``  the schema-2.0 checkpoint meta, the tensor index (name, shape, dtype) and the sha256 of the .h5.
A bundle directory holds the members, the dN/dX-to-w_c map ``wc_map.json`` (wc_map.py, with its own lineage record)
and ``manifest.json`` (the strict manifest of ``safe_io``), and nothing else. The h5/2 manifest's keys are exactly
format, members, files (every other file, pinned by sha256 and size), provenance, wc_map (the map's file name) and
lineage: the bundle's lineage block (``make_lineage_block``): the training cache (sha256, size, the digest of its lineage
record, its settings, its row origins and the ``carried_validated`` label of carried rows), the digest of its catalogue
set, the finder ids, the code commit that trained the members, the export's code identity, the w_c map (sha256, the
digest of its record), and ``release_eligible`` with the reasons when it is not.

Loading with a manifest digest (the release path, ``load_ensemble`` / ``load_bundle`` / ``load_release``) requires h5/2,
a well-formed lineage block whose eligibility recomputes to true under the release finder allow-list, the w_c map the
block names (its record eligible and fitted to the block's cache), every member trained on the block's cache
(``cache_sha256`` and ``cache_lineage_sha256``) at the block's commit from a clean tree, and every member a production
member (``eligibility.production_recipe_reasons``: its recipe declares the full training set, no reserved held-out or
fold key, no research marking), and the w_c map fitted on the whole training cache (its ``delta_c.n_blocks`` equals the
block's ``cache.n_groups``: ``wc_map_production_reasons``). Every reason carries a stable code
(``eligibility.reason_code``: ``LIN_*`` for the block, ``TIE_*`` for the members' ties, the production rule's codes). A
bundle whose provenance carries a research marking (``provenance.research``: cross-validation folds, full-simulation
research models) is refused by every release loader; it loads only through ``load_research`` (anchored, with a purpose),
which returns a ``ResearchBundle``, never a ``ReleaseBundle``. ``expected_manifest_sha256=None`` loads an unanchored
development bundle: the lineage is not enforced, a warning is issued, and the returned metas are marked
(``MemberMetas.anchored`` False, ``kind`` "dev"). The block is a declaration pinned by the manifest digest: each builder
verified its inputs when it ran; the loader checks that the declaration is complete, consistent and eligible.

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

import copy
import hashlib
import io
import json
import os
import re
import warnings
from typing import NamedTuple

import numpy as np

from . import eligibility as E
from . import safe_io as S

FORMAT = "hcd_priya.h5/2"                    # the release format (bundles with a lineage block and the w_c map)
FORMAT_HISTORICAL = "hcd_priya.h5/1"         # bundles written before bundles carried lineage: historical loaders only
FORMATS = (FORMAT, FORMAT_HISTORICAL)
EXT = ".h5"
WC_MAP_KEY = "wc_map"                        # manifest key naming the w_c map file
WC_MAP_FILE = "wc_map.json"
LINEAGE_KEY = "lineage"                      # manifest key of the lineage block
LINEAGE_FORMAT = "hcd_priya.bundle_lineage/1"
CARRIED_LABEL = "carried_validated"
CATALOGUE_SET_RULE = ("sha256 of the canonical JSON (sorted keys, no spaces) of the sorted [role, sha256] pairs of the "
                      "training cache record's inputs of kind catalog")
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
def export_member(prefix, model, meta, norm, *, fmt=FORMAT):
    """Write ``<prefix>.h5`` and ``<prefix>.json`` for one member. Never touches existing files; the .h5 is written
    through the temporary file's own descriptor and published by hard link only if that name still holds it.
    ``fmt``: ``FORMAT`` (h5/2, the release format; the meta's ``cache_path`` must be a plain file name or None, so a
    bundle never carries a private path) or ``FORMAT_HISTORICAL`` (h5/1, for reproducing historical bundles)."""
    h5py = _h5()
    if fmt not in FORMATS:
        raise SafeFormatError(f"unknown member format {fmt!r}")
    hp, js = prefix + EXT, prefix + ".json"
    for p in (hp, js):
        if os.path.lexists(p):
            raise SafeFormatError(f"{p} exists: export never overwrites")
    S._validate_meta(meta, prefix)                             # includes: plain finite JSON, before anything is written
    if fmt == FORMAT and meta.get("cache_path") is not None and not _plain_name(meta["cache_path"]):
        raise SafeFormatError(f"{prefix}: meta cache_path {meta['cache_path']!r} is a path; an h5/2 member records "
                              "the cache's file name only")
    tensors = {**S.model_tensors(model), **S.norm_tensors(norm)}
    for n, a in tensors.items():
        if S._code(a.dtype) != "F64":
            raise SafeFormatError(f"tensor {n!r} is {a.dtype}: this container stores float64 only")
    S._require_finite(tensors, prefix)

    def writer(fh):
        with h5py.File(fh, "w", libver=_LIBVER) as f:
            f.attrs["format"] = fmt
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
    doc = {"format": fmt, "meta": meta, "tensors": index, "h5_sha256": h5_sha,
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


def read_member_arrays(raw, where, want_model, norm_check, *, fmt=FORMAT):
    """Strictly parse the anchored .h5 bytes ``raw``. ``want_model`` = [(name, shape, dtype)] of the network derived
    from code; ``norm_check(tensors)`` validates the normalizer structure. Both are checked on metadata BEFORE any data
    is read (the normalizer through zero-memory views of the declared shapes). Returns ``{tensor name: array}``."""
    h5py = _h5()
    try:
        with h5py.File(io.BytesIO(raw), "r") as f:
            if f.userblock_size:
                raise SafeFormatError(f"{where}: a user block is not allowed")
            got = None
            if len(f.attrs) == 1 and list(f.attrs.keys()) == ["format"]:   # a count, then one name; no value yet
                aid = f.attrs.get_id("format")
                if aid.shape == () and aid.get_type().get_class() == h5py.h5t.STRING:
                    got = f.attrs["format"]                        # one scalar string, at most the file's size
            if not isinstance(got, str) or got != fmt:
                raise SafeFormatError(f"{where}: root attributes must be exactly a scalar string format = {fmt}")
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


def load_member(prefix, *, expected_sha256=None, expected_json_sha256=None, fmt=FORMAT):
    """Load one member exported by ``export_member``: a low-level member read, NOT a bundle load and never a production
    model (no lineage, no production or research rule; it returns no bundle object). Returns ``(model, meta, norm)``;
    refuses everything ``safe_io.load_member`` refuses plus the HDF5-specific structures listed in the module
    docstring. The .h5 is checked against its own sidecar; without ``expected_sha256`` / ``expected_json_sha256`` (the
    manifest's digests, which ``load_bundle`` passes) nothing outside the member anchors the pair, so a release loads
    through ``load_ensemble`` (or ``load_bundle`` / ``load_release``) with the published manifest digest, and a research
    bundle through ``load_research``."""
    import jax
    import jax.numpy as jnp

    for name, v in (("expected_sha256", expected_sha256), ("expected_json_sha256", expected_json_sha256)):
        if v is not None:                                    # an exact str only (safe_io.exact_hex64)
            S.require_exact_hex64(v, f"load_member: {name}")
    js = prefix + ".json"
    raw_js = S.read_regular(js, S.MAX_JSON_BYTES)
    if expected_json_sha256 is not None and hashlib.sha256(raw_js).hexdigest() != expected_json_sha256:
        raise SafeFormatError(f"{js}: sha256 does not match the manifest")
    doc = S._json_loads(raw_js, js)
    need = ("format", "meta", "tensors", "h5_sha256", "h5_file")
    if not isinstance(doc, dict) or any(k not in doc for k in need):
        raise SafeFormatError(f"{js}: sidecar lacks {[k for k in need if not isinstance(doc, dict) or k not in doc]}")
    if fmt not in FORMATS:
        raise SafeFormatError(f"unknown member format {fmt!r}")
    if doc["format"] != fmt:
        raise SafeFormatError(f"{js}: format {doc['format']!r}, this loader reads {fmt}"
                              + (" (a historical member: load_ensemble_historical)" if doc["format"] ==
                                 FORMAT_HISTORICAL else ""))
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
    tensors = read_member_arrays(raw, hp, want, norm_check, fmt=fmt)
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


# --------------------------------------------------------------------------------------------- lineage block
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_PLAIN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")
_BLOCK_KEYS = {"format", "cache", "catalogues", "finder_ids", "code_commit", "export_code", "wc_map", "n_members",
               "release_eligible", "ineligible_reasons"}
_CACHE_KEYS = {"sha256", "size", "lineage_record_sha256", "kind", "kind_version", "release_eligible", "settings",
               "n_groups", "rows", "carried"}
_CARRIED_KEYS = {"label", "fields", "source_sha256", "source_size", "n_groups", "validation_all_passed",
                 "validation_summary_sha256", "promoted"}
_WC_KEYS = {"path", "sha256", "lineage_record_sha256", "release_eligible", "cache_sha256"}


def _plain_name(x):
    return isinstance(x, str) and bool(_PLAIN.match(x))


def _hex64(x):
    return isinstance(x, str) and bool(_HEX64.match(x))


def _exact_hex64(x):
    """A caller-given anchor digest: 64 lowercase hex digits as an EXACT str. A str subclass can override == and !=
    and pass every later digest comparison, so it is refused, not compared (as the MF and release digests are). The
    one rule for every caller-given digest: ``safe_io.exact_hex64``."""
    return S.exact_hex64(x)


def _release_finder_ids():
    from ..catalog import RELEASE_FINDER_IDS
    return RELEASE_FINDER_IDS


def catalogue_set_sha256(cache_record):
    """``(n, sha256)`` of the training cache record's catalogue inputs (``CATALOGUE_SET_RULE``)."""
    from .. import lineage as L
    pairs = sorted([i["role"], i["sha256"]] for i in cache_record["inputs"] if i.get("kind") == "catalog")
    return len(pairs), hashlib.sha256(L.canonical_json(pairs).encode("ascii")).hexdigest()


def make_lineage_block(*, cache_record, cache_identity, wc_map, member_metas, export_code, allowed_finder_ids=None,
                       research=None):
    """The bundle's lineage block, from the verified training-cache record (``tau0_cache.read_cache``), the cache
    file's ``{sha256, size}``, the parsed w_c map (``wc_map.WcMap``), the members' metas and the export's code identity.
    ``release_eligible`` is computed here (``lineage_block_reasons`` plus ``member_reasons``: the members' ties and the
    production rule), never passed in, with the cache record's simulation names (``simulation_sets/1`` judged
    exactly). ``research`` (the bundle's ``provenance.research``, ``eligibility.check_research_marking``) adds the
    reason ``[RESEARCH_BUNDLE] research bundle: <purpose>``, so a research block is never release-eligible, and the
    member admission (``eligibility.research_member_reasons`` with the cache's names): any ``RES_*`` reason recorded
    here is enforced by the research loader."""
    from .. import lineage as L
    if research is not None:
        try:
            E.check_research_marking(research)
        except E.EligibilityError as e:
            raise SafeFormatError(str(e)) from None
    from ..tau0_cache import ROWS_CARRIED
    allowed = _release_finder_ids() if allowed_finder_ids is None else frozenset(allowed_finder_ids)
    rec = cache_record
    groups = rec["product"].get("groups") or []
    n_carried = sum(1 for g in groups if g.get("rows") == ROWS_CARRIED)
    car = rec["product"].get("carried") or {}
    val = rec.get("validation") or {}
    carried = None if not n_carried else {
        "label": CARRIED_LABEL, "fields": list(car.get("fields") or []), "source_sha256": car.get("source_sha256"),
        "source_size": car.get("source_size"), "n_groups": n_carried,
        "validation_all_passed": val.get("all_passed") is True, "validation_summary_sha256": val.get("summary_sha256"),
        "promoted": copy.deepcopy(car.get("promoted"))}
    n_cat, cat_sha = catalogue_set_sha256(rec)
    commits = {m.get("git_sha") for m in member_metas}
    block = L.canonicalise({
        "format": LINEAGE_FORMAT,
        "cache": {"sha256": cache_identity["sha256"], "size": int(cache_identity["size"]),
                  "lineage_record_sha256": L.record_sha256(rec), "kind": rec["kind"],
                  "kind_version": rec["kind_version"], "release_eligible": rec["release_eligible"] is True,
                  "settings": rec.get("settings") or {}, "n_groups": len(groups),
                  "rows": {"built": len(groups) - n_carried, CARRIED_LABEL: n_carried}, "carried": carried},
        "catalogues": {"n": n_cat, "sha256": cat_sha, "rule": CATALOGUE_SET_RULE},
        "finder_ids": sorted(rec["finder_ids"]),
        "code_commit": commits.pop() if len(commits) == 1 else None,
        "export_code": {"commit": export_code.get("commit"), "dirty": export_code.get("dirty")},
        "wc_map": {"path": WC_MAP_FILE, "sha256": wc_map.sha256, "lineage_record_sha256": L.record_sha256(wc_map.record),
                   "release_eligible": wc_map.record["release_eligible"] is True, "cache_sha256": wc_map.cache_sha256},
        "n_members": len(member_metas)})
    sims = cache_simulations(rec)
    why = lineage_block_reasons(block, allowed) + member_reasons(block, member_metas, sims=sims)
    if research is not None:
        why.append(E.research_reason(research["purpose"]))
        why += E.research_member_reasons(member_metas, research, sims=sims)
    else:
        why += wc_map_production_reasons(block, wc_map)
    block["release_eligible"] = not why
    block["ineligible_reasons"] = why
    return check_lineage_block(block)


def lineage_block_reasons(block, allowed_finder_ids):
    """The coded reasons (empty: eligible) why the block, read on its own, does not describe a release bundle. Every
    code starts with ``LIN_`` (``eligibility.code_class`` "lineage")."""
    from ..tau0_cache import CACHE_VERSION, CARRIED_FIELDS, KIND_VERSION
    R = E.reason
    why = []
    allowed = frozenset(allowed_finder_ids)
    fids = block.get("finder_ids") or []
    if not fids:
        why.append(R("LIN_FINDER_NONE", "no finder id"))
    bad = sorted(f for f in fids if f not in allowed)
    if bad:
        why.append(R("LIN_FINDER_NOT_ALLOWED", f"finder ids not in the release allow-list: {bad}"))
    c = block.get("cache") or {}
    if c.get("kind") != "tau0_cache" or c.get("release_eligible") is not True:
        why.append(R("LIN_CACHE_NOT_ELIGIBLE", "the training cache is not a release-eligible tau0_cache"))
    if c.get("kind_version") != KIND_VERSION or (c.get("settings") or {}).get("cache_version") != CACHE_VERSION:
        why.append(R("LIN_CACHE_VERSION", f"the training cache is not a version {CACHE_VERSION} tau0_cache record of "
                                          f"kind version {KIND_VERSION}"))
    rows = c.get("rows") or {}
    if c.get("n_groups") != rows.get("built", 0) + rows.get(CARRIED_LABEL, 0):
        why.append(R("LIN_CACHE_GROUPS", "the cache's group count is not the sum of its built and carried groups"))
    cat = block.get("catalogues") or {}
    if cat.get("n", 0) < 1:
        why.append(R("LIN_CATALOGUE_NONE", "the training cache names no catalogue"))
    if cat.get("rule") != CATALOGUE_SET_RULE:
        why.append(R("LIN_CATALOGUE_RULE", "the catalogue-set digest is not made by this format's rule"))
    if (c.get("rows") or {}).get(CARRIED_LABEL, 0):
        car = c.get("carried") or {}
        prom = car.get("promoted") or {}
        if car.get("label") != CARRIED_LABEL or car.get("fields") != list(CARRIED_FIELDS) \
                or not _hex64(car.get("source_sha256")) or car.get("validation_all_passed") is not True:
            why.append(R("LIN_CARRIED_LABEL", "carried rows without the carry's label, fields, source and passed "
                                              "validation"))
        if not (isinstance(prom, dict) and prom.get("passed") is True and _hex64(prom.get("comparison_sha256"))):
            why.append(R("LIN_CARRIED_PROMOTED", "carried rows not promoted (the fresh-row comparison has not "
                                                 "passed)"))
    if not (isinstance(block.get("code_commit"), str) and _HEX40.match(block["code_commit"])):
        why.append(R("LIN_CODE_COMMIT", "the members' code commit is unknown or not one commit"))
    ex = block.get("export_code") or {}
    if not (isinstance(ex.get("commit"), str) and _HEX40.match(ex["commit"])) or ex.get("dirty") is not False:
        why.append(R("LIN_EXPORT_CODE", "the export ran from an unknown commit or a dirty tree"))
    w = block.get("wc_map") or {}
    if w.get("release_eligible") is not True:
        why.append(R("LIN_WCMAP_NOT_ELIGIBLE", "the w_c map is not release-eligible"))
    if w.get("cache_sha256") != c.get("sha256"):
        why.append(R("LIN_WCMAP_CACHE", "the w_c map was fitted to another cache"))
    if w.get("path") != WC_MAP_FILE:
        why.append(R("LIN_WCMAP_PATH", f"the w_c map is not {WC_MAP_FILE}"))
    if not isinstance(block.get("n_members"), int) or block["n_members"] < 1:
        why.append(R("LIN_NO_MEMBER", "no member"))
    return why


def member_tie_reasons(block, metas):
    """The coded reasons (``TIE_*``) why the members are not the ones the block describes: each trained on the block's
    cache (sha256 and lineage-record digest) at the block's commit from a clean tree, recording a file name, not a
    path, and as many as the block states."""
    R = E.reason
    why = []
    c = block.get("cache") or {}
    if len(metas) != block.get("n_members"):
        why.append(R("TIE_COUNT", f"{len(metas)} members, the block states {block.get('n_members')}"))
    for i, m in enumerate(metas):
        if m.get("cache_sha256") != c.get("sha256"):
            why.append(R("TIE_CACHE", f"member {i} was trained on another cache"))
        if m.get("cache_lineage_sha256") != c.get("lineage_record_sha256"):
            why.append(R("TIE_CACHE_RECORD", f"member {i} does not name the cache's lineage record"))
        if m.get("git_sha") != block.get("code_commit") or not (isinstance(m.get("git_sha"), str)
                                                                and _HEX40.match(m["git_sha"])):
            why.append(R("TIE_COMMIT", f"member {i} was trained at another or an unknown commit"))
        if m.get("code_dirty") is not False:
            why.append(R("TIE_DIRTY", f"member {i} was trained from a tree not known to be clean"))
        if m.get("cache_path") is not None and not _plain_name(m["cache_path"]):
            why.append(R("TIE_CACHE_PATH", f"member {i} records a cache path, not a file name"))
    return why


def member_production_reasons(metas, *, sims=None, n_sims=None):
    """The coded production-rule reasons of every member (``eligibility.production_recipe_reasons``: a recipe that
    declares the full training set, no reserved held-out or fold key, no simulation subset named anywhere, no research
    marking). ``sims``: the cache's simulation names (the block's writer); ``n_sims``: its count (a load)."""
    return [r for i, m in enumerate(metas) for r in E.production_recipe_reasons(m, i, sims=sims, n_sims=n_sims)]


def wc_map_n_sims(wcm):
    """The number of simulations the w_c map's delta_c fit used (``delta_c.n_sims``, pinned by the manifest), or
    None. For a map that passes ``wc_map_production_reasons`` it is the training cache's simulation count."""
    v = ((getattr(wcm, "doc", None) or {}).get("delta_c") or {}).get("n_sims")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def cache_simulations(cache_record):
    """The simulation names of a training-cache lineage record's snapshot groups."""
    return sorted({str(g.get("sim")) for g in (cache_record.get("product") or {}).get("groups") or []})


def wc_map_production_reasons(block, wcm):
    """The coded reason (``WCMAP_NOT_FULL_CACHE``, a production-rule code) why the bundle's w_c map ``wcm`` (a parsed
    ``wc_map.WcMap``) is not a map fitted on the whole training cache the block names: its ``delta_c.n_blocks`` (the
    snapshot groups the delta_c fit used) differs from the block's ``cache.n_groups``. A cross-validation fold or any
    other subset refit counts fewer groups. Both numbers are pinned by the manifest digest. Fail-closed edge: a full
    cache holding a group without sightlines would be refused too, because the delta_c fit skips such a group."""
    n_blocks = ((getattr(wcm, "doc", None) or {}).get("delta_c") or {}).get("n_blocks")
    n_groups = (block.get("cache") or {}).get("n_groups")
    if not (isinstance(n_blocks, int) and n_blocks == n_groups):
        return [E.reason("WCMAP_NOT_FULL_CACHE", f"the w_c map was fitted on {n_blocks} snapshot groups, the training "
                                                 f"cache has {n_groups}: a fold or subset map is never a production "
                                                 "map")]
    return []


def member_reasons(block, metas, *, sims=None, n_sims=None):
    """The reasons why the members are not a production ensemble the block describes: ``member_tie_reasons`` plus
    ``member_production_reasons`` (``sims`` / ``n_sims`` as there). ``make_lineage_block`` (with the cache's names),
    every release load and ``predictor.Bundle`` (with the map's simulation count) use it, so ``release_eligible``
    reflects both."""
    return member_tie_reasons(block, metas) + member_production_reasons(metas, sims=sims, n_sims=n_sims)


def check_lineage_block(block):
    """Refuse (SafeFormatError) a lineage block that is not well formed (keys, types, digests); returns it. Says
    nothing about eligibility (``lineage_block_reasons``)."""
    def bad(why):
        raise SafeFormatError(f"lineage block: {why}")
    if not isinstance(block, dict) or set(block) != _BLOCK_KEYS:
        bad(f"keys {sorted(block) if isinstance(block, dict) else type(block).__name__}, expected {sorted(_BLOCK_KEYS)}")
    if block["format"] != LINEAGE_FORMAT:
        bad(f"format {block['format']!r}, expected {LINEAGE_FORMAT}")
    c = block["cache"]
    if not isinstance(c, dict) or set(c) != _CACHE_KEYS:
        bad("malformed cache entry")
    if not (_hex64(c["sha256"]) and _hex64(c["lineage_record_sha256"]) and S._is_int(c["size"]) and c["size"] >= 0
            and isinstance(c["settings"], dict) and isinstance(c["rows"], dict)
            and set(c["rows"]) == {"built", CARRIED_LABEL} and all(S._is_int(v) for v in c["rows"].values())
            and isinstance(c["release_eligible"], bool)):
        bad("malformed cache entry")
    if c["carried"] is not None and (not isinstance(c["carried"], dict) or set(c["carried"]) != _CARRIED_KEYS):
        bad("malformed carried entry")
    cat = block["catalogues"]
    if not (isinstance(cat, dict) and set(cat) == {"n", "sha256", "rule"} and S._is_int(cat["n"])
            and _hex64(cat["sha256"])):
        bad("malformed catalogues entry")
    f = block["finder_ids"]
    if not (isinstance(f, list) and all(isinstance(x, str) and x for x in f) and f == sorted(set(f))):
        bad("finder_ids must be a sorted list of distinct non-empty strings")
    if block["code_commit"] is not None and not isinstance(block["code_commit"], str):
        bad("code_commit must be a string or null")
    ex = block["export_code"]
    if not (isinstance(ex, dict) and set(ex) == {"commit", "dirty"}):
        bad("export_code must be {commit, dirty}")
    w = block["wc_map"]
    if not (isinstance(w, dict) and set(w) == _WC_KEYS and _hex64(w["sha256"]) and _hex64(w["lineage_record_sha256"])
            and isinstance(w["release_eligible"], bool)):
        bad("malformed wc_map entry")
    if not isinstance(block["release_eligible"], bool) or not (isinstance(block["ineligible_reasons"], list) and all(
            isinstance(r, str) for r in block["ineligible_reasons"])):
        bad("release_eligible must be a boolean and ineligible_reasons a list of strings")
    if block["release_eligible"] == bool(block["ineligible_reasons"]):
        bad("release_eligible disagrees with ineligible_reasons")
    return block


# --------------------------------------------------------------------------------------------- bundles
class UnanchoredBundleWarning(UserWarning):
    """A bundle was loaded without a manifest digest (a development load): nothing anchors it, lineage not enforced."""


class MemberMetas(list):
    """The members' metas, in member order (a list, as before), marked with how the bundle was loaded:
    ``kind`` ("release", "research", "dev" or "historical"; always set by the loader), ``release_verified`` (True only
    for a release load that enforced the lineage and the production rule), ``anchored`` (a manifest digest was given
    and matched), ``lineage_verified`` (the lineage block, the w_c map and the members' ties were enforced),
    ``manifest_sha256``, ``format``, ``lineage`` (the manifest's block, or None for h5/1) and ``historical_reason``
    (set by the historical loaders). A slice is a plain list: the marks never travel with part of the members. Only
    ``release_verified`` (equivalently ``kind == "release"``) means a production load: ``anchored`` and
    ``lineage_verified`` are also true after a research load."""

    def __init__(self, metas, *, anchored, lineage_verified, manifest_sha256, fmt, kind, lineage=None,
                 historical_reason=None, release_verified=False):
        if kind not in LOAD_KINDS:
            raise ValueError(f"kind must be one of {LOAD_KINDS}, got {kind!r}")
        super().__init__(metas)
        self.anchored, self.lineage_verified = bool(anchored), bool(lineage_verified)
        self.manifest_sha256, self.format = manifest_sha256, fmt
        self.lineage, self.historical_reason = lineage, historical_reason
        self.kind, self.release_verified = kind, release_verified is True and kind == "release"

    def __getitem__(self, i):
        out = super().__getitem__(i)
        return list(out) if isinstance(i, slice) else out


LOAD_KINDS = ("release", "research", "dev", "historical")


class ReleaseBundle(NamedTuple):
    """A loaded h5/2 bundle (``load_release``; the unanchored development path gives ``kind`` "dev"). ``kind`` is
    mandatory; ``release_verified`` defaults to the fail-closed False and is True only for a release load."""
    ensemble: object               # ensemble.EnsembleEmulator
    metas: MemberMetas
    norm: dict                     # member 0's normalizer (shared by every member)
    manifest: dict
    manifest_sha256: str
    lineage: dict
    wc_map: object                 # wc_map.WcMap
    anchored: bool
    bundle_dir: str
    kind: str                      # "release" or "dev"
    release_verified: bool = False


class ResearchBundle(NamedTuple):
    """A loaded research bundle (``load_research``): NOT a ``ReleaseBundle`` (a separate type, never accepted by
    ``predictor.Bundle``). ``purpose`` is the bundle's declared purpose, ``load_purpose`` the caller's, ``research``
    the manifest's ``provenance.research`` and ``split`` its summary (``eligibility.split_summary``)."""
    ensemble: object
    metas: MemberMetas
    norm: dict
    manifest: dict
    manifest_sha256: str
    lineage: dict
    wc_map: object
    anchored: bool
    bundle_dir: str
    kind: str                      # "research"
    purpose: str
    load_purpose: str
    research: dict
    split: dict
    release_verified: bool = False


def export_wc_map(bundle_dir, src_path, *, require_release_eligible=True):
    """Copy the w_c map at ``src_path`` into ``bundle_dir`` as ``WC_MAP_FILE`` (the bytes parsed and checked first;
    never overwritten). Returns the parsed map of the bytes written."""
    from . import wc_map as W
    raw = S.read_regular(str(src_path), W.MAX_BYTES)
    m = W.parse_wc_map(raw, str(src_path), require_release_eligible=require_release_eligible)
    dst = os.path.join(os.path.abspath(bundle_dir), WC_MAP_FILE)
    if os.path.lexists(dst):
        raise SafeFormatError(f"{dst} exists: export never overwrites")
    if S._write_exclusive(dst, [raw]) != m.sha256:
        raise SafeFormatError(f"{dst}: the bytes written are not the map's")
    return m


def _provenance_dict(provenance, where):
    if not isinstance(provenance, dict):
        raise SafeFormatError(f"{where}: provenance must be a JSON object, not {type(provenance).__name__}")
    return provenance


def research_marking_problems(lineage, provenance):
    """Why the block's research reason and ``provenance.research`` disagree (empty: consistent): a research bundle has
    both (the reason naming the marking's purpose), any other bundle neither."""
    reasons = lineage.get("ineligible_reasons") or []
    tagged = [r for r in reasons if E.reason_code(r) == "RESEARCH_BUNDLE"]
    if "research" not in provenance:
        return ["the block carries a research reason, the provenance no research marking"] if tagged else []
    try:
        rs = E.check_research_marking(provenance["research"])
    except E.EligibilityError as e:
        return [str(e)]
    out = []
    if tagged != [E.research_reason(rs["purpose"])]:
        out.append("the provenance is marked research, the block does not carry exactly its research reason "
                   f"{E.research_reason(rs['purpose'])!r}")
    if lineage.get("release_eligible") is not False:
        out.append("a research block must be stated ineligible")
    return out


def write_manifest(bundle_dir, member_prefixes, provenance, *, lineage):
    """The h5/2 ``manifest.json``: the members' files, the w_c map (``WC_MAP_FILE``, already in ``bundle_dir``), the
    provenance (a JSON object) and the lineage block (``make_lineage_block``), which must name that very map file. A
    research bundle carries ``provenance.research`` AND the block's research reason (``make_lineage_block(research=
    ...)``); a block and provenance that disagree are refused. Write the manifest last: before it exists nothing
    loads."""
    check_lineage_block(lineage)
    _provenance_dict(provenance, "write_manifest")
    why = research_marking_problems(lineage, provenance)
    if why:
        raise SafeFormatError("write_manifest: " + "; ".join(why))
    wc = os.path.join(os.path.abspath(bundle_dir), WC_MAP_FILE)
    if S._sha256_regular(wc)[0] != lineage["wc_map"]["sha256"]:
        raise SafeFormatError(f"{wc} is not the w_c map the lineage block names")
    return S.write_manifest(bundle_dir, member_prefixes, provenance, data_ext=EXT, fmt=FORMAT,
                            aux={WC_MAP_KEY: WC_MAP_FILE}, extra={LINEAGE_KEY: lineage})


def write_manifest_historical(bundle_dir, member_prefixes, provenance):
    """An h5/1 ``manifest.json`` (members only), for reproducing historical bundles; their members are exported with
    ``fmt=FORMAT_HISTORICAL``. An h5/1 bundle carries no research marking."""
    if "research" in _provenance_dict(provenance, "write_manifest_historical"):
        raise SafeFormatError("write_manifest_historical: an h5/1 bundle carries no research marking")
    return S.write_manifest(bundle_dir, member_prefixes, provenance, data_ext=EXT, fmt=FORMAT_HISTORICAL)


def _manifest_format(bundle_dir):
    try:
        man = S._json_loads(S.read_regular(os.path.join(os.path.abspath(bundle_dir), S.MANIFEST), S.MAX_JSON_BYTES),
                            S.MANIFEST)
    except SafeFormatError:
        return None
    return man.get("format") if isinstance(man, dict) else None


def _read_v2(bundle_dir, expected):
    if expected is not None and _manifest_format(bundle_dir) == FORMAT_HISTORICAL:
        raise SafeFormatError(f"{bundle_dir}: an {FORMAT_HISTORICAL} bundle (written before bundles carried lineage) "
                              f"is historical: it loads only through load_ensemble_historical(..., reason=...)")
    return S.read_manifest(bundle_dir, expected, data_ext=EXT, fmt=FORMAT, aux_keys=(WC_MAP_KEY,),
                           extra_keys=(LINEAGE_KEY,))


def verify_manifest(bundle_dir, *, expected_sha256=None, allow_extra=False):
    """Check an h5/2 bundle's manifest against the directory without loading it. Returns the member prefixes."""
    return [p for p, _, _ in S.read_manifest(bundle_dir, expected_sha256, allow_extra, data_ext=EXT, fmt=FORMAT,
                                             aux_keys=(WC_MAP_KEY,), extra_keys=(LINEAGE_KEY,))[2]]


def _same_network(bundle_dir, members):
    """Every member shares member 0's architecture, k grid, parameter box and normalizer."""
    _, meta0, norm0 = members[0]
    t0 = S.norm_tensors(norm0)
    for i, (_, meta, norm) in enumerate(members[1:], 1):
        if meta["arch_cfg"] != meta0["arch_cfg"] or meta["k_com_hmpc"] != meta0["k_com_hmpc"]:
            raise SafeFormatError(f"{bundle_dir}: member {i} has another architecture or k grid than member 0")
        if meta["param_limits"] != meta0["param_limits"]:
            raise SafeFormatError(f"{bundle_dir}: member {i} has another parameter box than member 0")
        t = S.norm_tensors(norm)
        if set(t) != set(t0) or any(t[n].shape != t0[n].shape or t[n].tobytes() != t0[n].tobytes() for n in t0):
            raise SafeFormatError(f"{bundle_dir}: member {i} does not share member 0's normalizer")


MODES = ("release", "research", "dev")


def _check_mode(mode, expected, purpose):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "dev":
        if expected is not None:
            raise SafeFormatError("a development load takes no manifest digest (expected_manifest_sha256=None); with "
                                  "a digest the bundle loads as a release (load_release) or research "
                                  "(load_research) bundle")
    elif not _exact_hex64(expected):
        raise SafeFormatError(f"a {mode} load needs the published manifest sha256 (64 lowercase hex digits, an exact "
                              "str); an unanchored development load is load_ensemble(..., "
                              "expected_manifest_sha256=None)")
    if mode == "research":
        if not (isinstance(purpose, str) and purpose.strip()):
            raise ValueError("a research load needs a non-empty purpose (why this research model is loaded)")
    elif purpose is not None:
        raise ValueError("purpose belongs to a research load (load_research) only")


def _research_block_problems(block, rs, allowed):
    """Research mode's lineage rules (v3 2.6, v3.1 E2): the block's own lineage reasons recompute to none; the stored
    block is ineligible, carries exactly the research reason of ``rs``'s purpose, every stored reason is coded, and
    none of them is a lineage or tie code (a stale block). The per-member production codes are not compared."""
    out = [f"the lineage does not establish a clean lineage: {r}" for r in lineage_block_reasons(block, allowed)]
    stored = block["ineligible_reasons"]
    if block["release_eligible"] is not False:
        out.append("the block is stated release-eligible")
    if [r for r in stored if E.reason_code(r) == "RESEARCH_BUNDLE"] != [E.research_reason(rs["purpose"])]:
        out.append(f"the block does not carry exactly the research reason {E.research_reason(rs['purpose'])!r}")
    uncoded = [r for r in stored if E.reason_code(r) is None]
    if uncoded:
        out.append(f"the block carries reasons without a code: {uncoded[:3]}")
    stale = [r for r in stored if E.code_class(r) in ("lineage", "tie")]
    if stale:
        out.append(f"the block carries lineage or tie reasons (a stale block): {stale[:3]}")
    return out


def _load_v2(bundle_dir, *, expected, mode, purpose=None):
    """Everything an h5/2 load does. ``mode`` (required): "release" (a 64-hex digest; the lineage, the members' ties
    and the production rule enforced; a research marking refused), "research" (a 64-hex digest and a purpose; the
    research marking required and checked, the lineage and the ties enforced, the members admitted by
    ``eligibility.research_member_reasons``), or "dev" (no digest; nothing enforced, a warning). Returns the bundle
    object (``ReleaseBundle`` or ``ResearchBundle``, decided here) and the members."""
    from .. import lineage as L
    from . import wc_map as W
    from .ensemble import EnsembleEmulator
    _check_mode(mode, expected, purpose)
    man, sha, entries, aux = _read_v2(bundle_dir, expected)
    prov = man["provenance"]
    if not isinstance(prov, dict):
        raise SafeFormatError(f"{bundle_dir}: the manifest's provenance is not a JSON object")
    block = check_lineage_block(man[LINEAGE_KEY])
    wc_path, wc_sha = aux[WC_MAP_KEY]
    try:
        wcm = W.parse_wc_map(S.read_regular(wc_path, W.MAX_BYTES), wc_path, expected_sha256=wc_sha,
                             require_release_eligible=mode != "dev")
    except W.WcMapError as e:
        raise SafeFormatError(str(e)) from None
    rs = None
    if mode == "release":
        if "research" in prov:
            raise SafeFormatError(f"{bundle_dir}: a research bundle (provenance.research) is never a production model; "
                                  "it loads only through safe_h5.load_research / predictor.load_research_bundle")
        why = lineage_block_reasons(block, _release_finder_ids())
        if why or block["release_eligible"] is not True or block["ineligible_reasons"]:
            raise SafeFormatError(f"{bundle_dir}: the bundle's lineage does not establish a clean release: "
                                  + "; ".join(why or block["ineligible_reasons"] or ["stated ineligible"]))
    elif mode == "research":
        if "research" not in prov:
            raise SafeFormatError(f"{bundle_dir}: not a research bundle (the provenance carries no research marking); "
                                  "a production bundle loads through load_release / predictor.load_bundle")
        try:
            rs = E.check_research_marking(prov["research"])
        except E.EligibilityError as e:
            raise SafeFormatError(f"{bundle_dir}: {e}") from None
        why = _research_block_problems(block, rs, _release_finder_ids())
        if why:
            raise SafeFormatError(f"{bundle_dir}: the research bundle's lineage block is refused: " + "; ".join(why))
        if rs["wc_map"]["n_sims"] != wcm.doc["delta_c"]["n_sims"]:
            raise SafeFormatError(f"{bundle_dir}: provenance.research.wc_map.n_sims {rs['wc_map']['n_sims']} is not "
                                  f"the map's {wcm.doc['delta_c']['n_sims']}")
    if mode != "dev":
        if wcm.sha256 != block["wc_map"]["sha256"] or L.record_sha256(wcm.record) != \
                block["wc_map"]["lineage_record_sha256"] or wcm.cache_sha256 != block["cache"]["sha256"]:
            raise SafeFormatError(f"{bundle_dir}: the w_c map is not the one the lineage block names")
    members = [load_member(p, expected_sha256=s, expected_json_sha256=j, fmt=FORMAT) for p, s, j in entries]
    metas = [m for _, m, _ in members]
    if mode == "release":
        why, wwhy = member_reasons(block, metas, n_sims=wc_map_n_sims(wcm)), wc_map_production_reasons(block, wcm)
        if why:
            raise SafeFormatError(f"{bundle_dir}: the members are not the ones the lineage block describes: "
                                  + "; ".join(why + wwhy))
        if wwhy:
            raise SafeFormatError(f"{bundle_dir}: the w_c map was not fitted on the whole training cache: "
                                  + "; ".join(wwhy))
    elif mode == "research":
        why = member_tie_reasons(block, metas)
        if why:
            raise SafeFormatError(f"{bundle_dir}: the members are not the ones the lineage block describes: "
                                  + "; ".join(why))
        why = E.research_member_reasons(metas, rs)
        if why:
            raise SafeFormatError(f"{bundle_dir}: the members are not admitted to this research bundle: "
                                  + "; ".join(why))
        why = [r for r in block["ineligible_reasons"] if E.code_class(r) == "admission"]
        if why:
            raise SafeFormatError(f"{bundle_dir}: the members are not admitted to this research bundle (refused when "
                                  "it was written, with the cache's simulation names): " + "; ".join(why))
    _same_network(bundle_dir, members)
    if mode == "dev":
        warnings.warn(f"{bundle_dir}: loaded WITHOUT a manifest digest (unanchored development bundle); its lineage "
                      "was not enforced" + ("; it is a RESEARCH bundle (provenance.research), never a production "
                                            "model" if "research" in prov else ""), UnanchoredBundleWarning,
                      stacklevel=3)
    anchored = mode != "dev"
    mm = MemberMetas(metas, anchored=anchored, lineage_verified=anchored, manifest_sha256=sha, fmt=FORMAT,
                     lineage=copy.deepcopy(block), kind=mode, release_verified=mode == "release")
    common = (EnsembleEmulator([m for m, _, _ in members]), mm, members[0][2], man, sha, copy.deepcopy(block), wcm,
              anchored, os.path.abspath(bundle_dir))
    if mode == "research":
        import logging
        logging.getLogger(__name__).info("research load of %s (%s): %s; not a production model", bundle_dir,
                                         purpose.strip(), rs["purpose"])
        rb = ResearchBundle(*common, kind="research", purpose=rs["purpose"], load_purpose=purpose.strip(),
                            research=copy.deepcopy(rs), split=E.split_summary(rs))
    else:
        rb = ReleaseBundle(*common, kind=mode, release_verified=mode == "release")
    return rb, members


def load_release(bundle_dir, *, expected_manifest_sha256):
    """The release load of an h5/2 production bundle, anchored on the published manifest digest (None is refused):
    the lineage checks of ``load_ensemble`` plus the parsed w_c map and the block. Refuses a research bundle and any
    member that is not a production member (``member_reasons``). Returns a ``ReleaseBundle`` (``kind`` "release",
    ``release_verified`` True)."""
    if not _exact_hex64(expected_manifest_sha256):
        raise SafeFormatError("load_release needs the published manifest sha256 (64 lowercase hex digits, an exact "
                              "str); an unanchored development load is load_ensemble(..., "
                              "expected_manifest_sha256=None)")
    return _load_v2(bundle_dir, expected=expected_manifest_sha256, mode="release")[0]


def load_research(bundle_dir, *, expected_manifest_sha256, purpose):
    """The research load of an h5/2 research bundle (``provenance.research``; a cross-validation fold model, a
    full-simulation research model): anchored on its manifest digest (64 lowercase hex digits), with ``purpose``
    (non-empty; logged at INFO, kept as ``load_purpose`` and recorded in every research prediction's provenance).
    Enforces the lineage, the members' ties, the research marking (its recorded content checks and map rule
    included) and the member admission (``eligibility.research_member_reasons``: one declared split). Returns a
    ``ResearchBundle``, never a ``ReleaseBundle``; a production bundle is refused here."""
    return _load_v2(bundle_dir, expected=expected_manifest_sha256, mode="research", purpose=purpose)[0]


def _mode(expected):
    return "dev" if expected is None else "release"


def load_ensemble(bundle_dir, *, expected_manifest_sha256):
    """The release entry point: an h5/2 bundle, every member sharing one network architecture, one k grid, one
    parameter box and one normalizer (a prediction applies member 0's normalizer to every member). With the published
    manifest digest the bundle must establish a clean lineage and be a production bundle (module docstring) or it is
    refused; ``None`` loads an unanchored development bundle, warns, and marks the metas (``MemberMetas.anchored``
    False, ``kind`` "dev"). An h5/1 bundle is refused: ``load_ensemble_historical``. A research bundle is refused with
    a digest: ``load_research``. Returns ``(ensemble.EnsembleEmulator, MemberMetas, norm)``."""
    rb, _ = _load_v2(bundle_dir, expected=expected_manifest_sha256, mode=_mode(expected_manifest_sha256))
    return rb.ensemble, rb.metas, rb.norm


def load_bundle(bundle_dir, *, expected_manifest_sha256):
    """The members of an h5/2 bundle, ``[(model, meta, norm)]``, after every check of ``load_ensemble`` (anchor keyword
    required; ``None`` loads an unanchored development bundle and warns)."""
    return _load_v2(bundle_dir, expected=expected_manifest_sha256, mode=_mode(expected_manifest_sha256))[1]


def _historical_reason(reason):
    if not (isinstance(reason, str) and reason.strip()):
        raise ValueError("a historical bundle load needs a non-empty reason")
    return reason.strip()


def load_bundle_historical(bundle_dir, *, expected_manifest_sha256, reason):
    """The members of an h5/1 bundle (written before bundles carried lineage), ``[(model, meta, norm)]``: every format
    check of the h5/1 loader, no lineage. ``reason`` (non-empty) is logged; nothing loaded this way is a release
    model."""
    import logging
    why = _historical_reason(reason)
    logging.getLogger(__name__).warning("historical load of the bundle %s (%s): not a release model", bundle_dir, why)
    return [load_member(p, expected_sha256=s, expected_json_sha256=j, fmt=FORMAT_HISTORICAL)
            for p, s, j in S._manifest_entries(bundle_dir, expected_manifest_sha256, data_ext=EXT,
                                               fmt=FORMAT_HISTORICAL)]


def load_ensemble_historical(bundle_dir, *, expected_manifest_sha256, reason):
    """``load_ensemble`` for an h5/1 bundle: ``(EnsembleEmulator, MemberMetas, norm)`` with the metas marked
    historical (``historical_reason``) and ``lineage_verified`` False."""
    from .ensemble import EnsembleEmulator
    members = load_bundle_historical(bundle_dir, expected_manifest_sha256=expected_manifest_sha256, reason=reason)
    _same_network(bundle_dir, members)
    sha = S.sha256_file(os.path.join(os.path.abspath(bundle_dir), S.MANIFEST))
    mm = MemberMetas([m for _, m, _ in members], anchored=expected_manifest_sha256 is not None,
                     lineage_verified=False, manifest_sha256=sha, fmt=FORMAT_HISTORICAL, kind="historical",
                     historical_reason=_historical_reason(reason))
    return EnsembleEmulator([m for m, _, _ in members]), mm, members[0][2]


def verify_manifest_historical(bundle_dir, *, expected_sha256=None, allow_extra=False):
    """Check an h5/1 bundle's manifest against the directory without loading it. Returns the member prefixes."""
    return [p for p, _, _ in S._manifest_entries(bundle_dir, expected_sha256, allow_extra, data_ext=EXT,
                                                 fmt=FORMAT_HISTORICAL)]


sha256_file = S.sha256_file
