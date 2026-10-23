"""Pickle-free release format for the emulator: format ``hcd_priya.safe/1`` (contract: docs/release_format.md).

Layout of one LF member ``<prefix>``:
  ``<prefix>.safetensors``  every array: the network leaves (``model<tree path>``) and the normalizer (``norm/<key>/...``;
                            an integer dict key is written ``#<int>``), in the standard safetensors layout (8-byte
                            little-endian header length, a JSON header giving each tensor's dtype, shape and byte range,
                            then the raw little-endian buffer). Written and read here with numpy only; the official
                            ``safetensors`` package reads the same files (tests/external).
  ``<prefix>.json``         the schema-2.0 checkpoint meta, the tensor index (name, shape, dtype), the .safetensors file
                            name and its sha256.
A bundle directory holds the members and ``manifest.json`` (format, member order, every file's sha256 and size,
provenance) and nothing else. Tables (MF corrections, grids) go to HDF5 with numeric datasets and scalar attributes only.

Loading never imports or calls anything named in a file: the whole safetensors header is validated (entries,
contiguous non-overlapping spans, no trailing bytes) before any tensor is copied out of the file's bytes; the network
shape is derived from reviewed code (``model.Emulator`` under ``equinox.filter_eval_shape``, nothing allocated) and an
architecture config checked against an allowlist; every leaf is matched by name, shape and dtype before the network is
assembled, and the normalizer must have the production structure, all float64. No pickle is read or written on this
path."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import struct

import numpy as np

FORMAT = "hcd_priya.safe/1"
# safetensors dtype codes this format writes; float64 stays float64 (never down-cast)
_DTYPES = {"F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "I64": np.dtype("<i8"), "I32": np.dtype("<i4"),
           "U8": np.dtype("|u1"), "BOOL": np.dtype("|b1")}
_CODES = {v: k for k, v in _DTYPES.items()}
_MAX_HEADER = 100_000_000                    # the official reader's limit
_MAX_NDIM = 8
ARCH_KEYS = ("in_dim", "n_k", "n_basis", "baseline_n_layers", "baseline_width")   # Emulator(**arch_cfg)
_ARCH_MAX = 1 << 16
_INT_SEGMENT = re.compile(r"#(0|-?[1-9][0-9]{0,17})\Z")        # canonical, fits int64
_MAX_DIM = 1 << 40
_PLAIN_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")
MANIFEST = "manifest.json"


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


def _json_loads(raw, where):
    """Strict JSON from bytes: UTF-8, no duplicate keys; every failure is a SafeFormatError."""
    try:
        return json.loads(bytes(raw).decode("utf-8", errors="strict"), object_pairs_hook=_no_duplicate_keys)
    except SafeFormatError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise SafeFormatError(f"{where}: unreadable JSON ({type(e).__name__}: {e})"[:300]) from None


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _code(dtype):
    return _CODES.get(dtype.newbyteorder("<") if dtype.byteorder == ">" else dtype)


def _publish(tmp, path):
    """Move ``tmp`` to ``path`` only if ``path`` does not exist (a hard link fails atomically on an existing target).
    An EEXIST is accepted only when ``path`` is the very same inode as ``tmp`` and not a symlink (an NFS retransmit can
    report EEXIST for our own new link); every other failure is a SafeFormatError."""
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            try:
                st_p, st_t = os.lstat(path), os.lstat(tmp)
            except OSError as e:
                raise SafeFormatError(f"{path} cannot be published ({e})") from None
            if stat.S_ISLNK(st_p.st_mode) or (st_p.st_dev, st_p.st_ino) != (st_t.st_dev, st_t.st_ino):
                raise SafeFormatError(f"{path} exists: never overwritten") from None
        except OSError as e:
            raise SafeFormatError(f"{path} cannot be published ({e})") from None
    finally:
        os.remove(tmp)


def _finalize_tmp(tmp):
    """fsync before publishing. The mode is the one any new file gets (0o666 under the umask and the directory's
    default ACLs): nothing is chmodded, so no permission or ACL entry wider than the directory's policy is added."""
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())


def _new_tmp(path, suffix=""):
    """A new temporary file next to ``path``, created exclusively and never through a symlink."""
    d = os.path.dirname(os.path.abspath(path))
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(100):
        tmp = os.path.join(d, f".tmp_{secrets.token_hex(8)}{suffix}")
        try:
            return os.open(tmp, flags, 0o666), tmp
        except FileExistsError:
            continue
        except OSError as e:
            raise SafeFormatError(f"{path} cannot be created ({e})") from None
    raise SafeFormatError(f"{path}: no free temporary name")


def _write_exclusive(path, chunks):
    fd, tmp = _new_tmp(path)
    try:
        with os.fdopen(fd, "wb") as f:
            for c in chunks:
                f.write(c)
        _finalize_tmp(tmp)
    except BaseException:
        os.remove(tmp)
        raise
    _publish(tmp, path)


def write_safetensors(path, tensors, metadata=None):
    """Write ``{name: array}`` to a NEW file ``path`` in the safetensors layout. Arrays keep their dtype."""
    if not tensors:
        raise SafeFormatError("no tensors to write")
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        if not isinstance(name, str) or not name or name == "__metadata__":
            raise SafeFormatError(f"invalid tensor name {name!r}")
        a = np.asarray(tensors[name])
        code = _code(a.dtype)
        if code is None or a.ndim > _MAX_NDIM:
            raise SafeFormatError(f"tensor {name!r}: dtype {a.dtype} / ndim {a.ndim} is not supported")
        buf = np.ascontiguousarray(a, dtype=_DTYPES[code]).tobytes()
        header[name] = {"dtype": code, "shape": [int(s) for s in a.shape], "data_offsets": [offset, offset + len(buf)]}
        chunks.append(buf)
        offset += len(buf)
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise SafeFormatError("metadata must map str to str")
        header["__metadata__"] = dict(metadata)
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    _write_exclusive(path, [struct.pack("<Q", len(hb)), hb] + chunks)


def parse_safetensors(raw, where="<bytes>"):
    """Parse safetensors bytes strictly. Returns ``(tensors, metadata)``; every array is a writable copy."""
    if len(raw) < 8:
        raise SafeFormatError(f"{where}: truncated (no header length)")
    (n,) = struct.unpack("<Q", raw[:8])
    if n < 2 or n > _MAX_HEADER or 8 + n > len(raw):
        raise SafeFormatError(f"{where}: header length {n} is invalid for a {len(raw)}-byte file")
    if raw[8:9] != b"{":
        raise SafeFormatError(f"{where}: header does not start with '{{'")
    header = _json_loads(raw[8:8 + n], where)
    if not isinstance(header, dict):
        raise SafeFormatError(f"{where}: header is not a JSON object")
    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise SafeFormatError(f"{where}: __metadata__ must map str to str")
    buf = memoryview(raw)[8 + n:]
    spans, plan = [], []
    for name, e in header.items():                     # pass 1: validate every entry; nothing is copied yet
        if not name:
            raise SafeFormatError(f"{where}: empty tensor name")
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            raise SafeFormatError(f"{where}: tensor {name!r} has a malformed entry")
        if not isinstance(e["dtype"], str) or e["dtype"] not in _DTYPES:
            raise SafeFormatError(f"{where}: tensor {name!r} has unsupported dtype {e['dtype']!r}")
        shape, off = e["shape"], e["data_offsets"]
        if (not isinstance(shape, list) or len(shape) > _MAX_NDIM
                or not all(_is_int(s) and 0 <= s < _MAX_DIM for s in shape)
                or not isinstance(off, list) or len(off) != 2 or not all(_is_int(o) for o in off)):
            raise SafeFormatError(f"{where}: tensor {name!r} has a malformed shape or offsets")
        a, b = off
        dt = _DTYPES[e["dtype"]]
        size = math.prod(shape) * dt.itemsize                 # Python ints: no overflow
        if not 0 <= a <= b <= len(buf) or b - a != size:
            raise SafeFormatError(f"{where}: tensor {name!r} offsets {off} do not match its shape/dtype or the file")
        spans.append((a, b, name))
        plan.append((name, a, b, dt, shape))
    spans.sort()
    pos = 0
    for a, b, name in spans:
        if a != pos:
            raise SafeFormatError(f"{where}: byte buffer has a gap or an overlap at tensor {name!r}")
        pos = b
    if pos != len(buf):
        raise SafeFormatError(f"{where}: {len(buf) - pos} trailing bytes not owned by any tensor")
    out = {}
    for name, a, b, dt, shape in plan:                 # pass 2: the spans tile the buffer exactly; copy each once
        try:
            out[name] = np.frombuffer(buf[a:b], dtype=dt).reshape(shape).copy()
        except ValueError as e:
            raise SafeFormatError(f"{where}: tensor {name!r} cannot take shape {shape} ({e})") from None
    return out, meta


def read_safetensors(path, expected_sha256=None):
    """Read and parse a safetensors file; with ``expected_sha256`` the SAME bytes that are parsed are hashed first."""
    with open(path, "rb") as f:
        raw = f.read()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SafeFormatError(f"{path}: sha256 does not match the expected digest")
    return parse_safetensors(raw, path)


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
        if isinstance(k, int) and not -10 ** 18 < k < 10 ** 18:
            raise SafeFormatError(f"normalizer key {k!r} under {prefix!r} has more than 18 digits")
        name = f"{prefix}/{'#%d' % k if isinstance(k, int) else k}"
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
    """Inverse of ``norm_tensors`` over the names under ``norm/``; refuses names it would not produce itself
    (non-canonical ``#`` segments, empty segments, a leaf that is also a subtree)."""
    root = {}
    names = sorted(n for n in tensors if n.startswith("norm/"))
    for name in names:
        segs = name.split("/")[1:]
        keys = []
        for s in segs:
            if s.startswith("#"):
                if not _INT_SEGMENT.match(s):
                    raise SafeFormatError(f"normalizer name {name!r}: integer segment {s!r} is not canonical")
                keys.append(int(s[1:]))
            elif s:
                keys.append(s)
            else:
                raise SafeFormatError(f"normalizer name {name!r} has an empty segment")
        node = root
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise SafeFormatError(f"normalizer name {name!r} nests under a leaf")
        if keys[-1] in node:
            raise SafeFormatError(f"normalizer name {name!r} collides with another tensor")
        node[keys[-1]] = tensors[name]
    if set(norm_tensors(root)) != set(names):
        raise SafeFormatError("normalizer names do not round-trip")
    return root


def _validate_norm(norm, n_k, n_cddf, n_dndx):
    """The production normalizer structure (train.fit_target_norm), every leaf float64."""
    def leaf(path, a, shape):
        if not isinstance(a, np.ndarray) or a.dtype != np.dtype("<f8") or a.shape != shape:
            got = (a.dtype, a.shape) if isinstance(a, np.ndarray) else type(a).__name__
            raise SafeFormatError(f"normalizer {path}: {got}, expected float64 {shape}")

    def keys(path, d, want):
        if not isinstance(d, dict) or set(d) != set(want):
            raise SafeFormatError(f"normalizer {path}: keys {sorted(map(str, d)) if isinstance(d, dict) else d!r}, "
                                  f"expected {sorted(want)}")
    keys("norm", norm, ("f_nhi", "dndx", "P_filt", "delta"))
    for top, shape in (("f_nhi", (n_cddf,)), ("dndx", (n_dndx,)), ("delta", (3, n_k))):
        keys(f"norm/{top}", norm[top], ("mean", "std"))
        for k in ("mean", "std"):
            leaf(f"norm/{top}/{k}", norm[top][k], shape)
    pf = norm["P_filt"]
    keys("norm/P_filt", pf, ("mu_marg", "sig_marg", "sig_cosmo", "cell_mean"))
    for k in ("mu_marg", "sig_marg", "sig_cosmo"):
        leaf(f"norm/P_filt/{k}", pf[k], (4, n_k))
    cm = pf["cell_mean"]
    if not isinstance(cm, dict) or not cm or not all(_is_int(k) for k in cm):
        raise SafeFormatError("normalizer norm/P_filt/cell_mean must be a non-empty dict with integer keys")
    for k, a in cm.items():
        leaf(f"norm/P_filt/cell_mean/#{k}", a, (4, n_k))


# --------------------------------------------------------------------------------------------- members
def _validate_arch(arch):
    if not isinstance(arch, dict) or set(arch) != set(ARCH_KEYS):
        raise SafeFormatError(f"arch_cfg keys {sorted(arch) if isinstance(arch, dict) else arch!r} are not the "
                              f"allowlist {sorted(ARCH_KEYS)}")
    for k in ARCH_KEYS:
        if k == "n_basis" and arch[k] is None:                   # train._arch_cfg_from_model: no basis head
            continue
        if not _is_int(arch[k]) or not 0 < arch[k] <= _ARCH_MAX:
            raise SafeFormatError(f"arch_cfg[{k!r}] = {arch[k]!r} is not an int in [1, {_ARCH_MAX}]")


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
    if not _is_int(meta.get("seed")) or not 0 <= meta["seed"] < 2 ** 32:
        raise SafeFormatError(f"{where}: seed must be an int in [0, 2**32)")


def _abstract_network(arch, seed):
    """(treedef, [(name, shape, dtype)]) of ``Emulator(**arch)``, from abstract evaluation: nothing is allocated."""
    import equinox as eqx
    import jax

    from .model import Emulator
    abstract = eqx.filter_eval_shape(lambda: Emulator(**arch, key=jax.random.PRNGKey(seed)))
    paths, treedef = jax.tree_util.tree_flatten_with_path(abstract)
    return treedef, [("model" + jax.tree_util.keystr(p), tuple(x.shape), np.dtype(x.dtype)) for p, x in paths]


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
    _write_exclusive(js, [json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")])
    return st, js


def load_member(prefix, *, expected_sha256=None, expected_json_sha256=None):
    """Load one member exported by ``export_member``. Returns ``(model, meta, norm)``.

    ``expected_*`` are the bundle manifest's digests of the two files (``load_bundle`` passes them); the bytes that
    are hashed are the bytes that are parsed. Refuses: another format version or checkpoint schema, an architecture
    outside the allowlist, a digest mismatch, a tensor file other than ``<prefix>.safetensors``, names outside
    ``model``/``norm/``, missing or extra tensors, any shape or dtype difference from the network derived from code,
    and a normalizer without the production structure or not float64."""
    import jax
    import jax.numpy as jnp

    js = f"{prefix}.json"
    with open(js, "rb") as f:
        raw_js = f.read()
    if expected_json_sha256 is not None and hashlib.sha256(raw_js).hexdigest() != expected_json_sha256:
        raise SafeFormatError(f"{js}: sha256 does not match the manifest")
    doc = _json_loads(raw_js, js)
    need = ("format", "meta", "tensors", "safetensors_sha256", "safetensors_file")
    if not isinstance(doc, dict) or any(k not in doc for k in need):
        raise SafeFormatError(f"{js}: sidecar lacks {[k for k in need if not isinstance(doc, dict) or k not in doc]}")
    if doc["format"] != FORMAT:
        raise SafeFormatError(f"{js}: format {doc['format']!r}, this loader reads {FORMAT}")
    meta = doc["meta"]
    _validate_meta(meta, js)
    if doc["safetensors_file"] != os.path.basename(prefix) + ".safetensors":
        raise SafeFormatError(f"{js}: safetensors_file {doc['safetensors_file']!r} is not "
                              f"{os.path.basename(prefix)}.safetensors")
    st = f"{prefix}.safetensors"
    with open(st, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != doc["safetensors_sha256"] or (expected_sha256 is not None and digest != expected_sha256):
        raise SafeFormatError(f"{st}: sha256 does not match {js}" + (" or the manifest" if expected_sha256 else ""))
    tensors, smeta = parse_safetensors(raw, st)
    if smeta.get("format") != FORMAT:
        raise SafeFormatError(f"{st}: format {smeta.get('format')!r}, expected {FORMAT}")
    index = doc["tensors"]
    if not isinstance(index, dict) or set(index) != set(tensors):
        raise SafeFormatError(f"{st}: tensor names differ from the JSON index")
    for n, a in tensors.items():
        e = index[n]
        if not isinstance(e, dict) or e.get("shape") != list(a.shape) or e.get("dtype") != _CODES.get(a.dtype):
            raise SafeFormatError(f"{st}: tensor {n!r} does not match its JSON index entry")
    stray = [n for n in tensors if not (n.startswith("model.") or n.startswith("model[") or n.startswith("norm/"))]
    if stray:
        raise SafeFormatError(f"{st}: tensors outside model/norm: {sorted(stray)[:5]}")
    treedef, want = _abstract_network(meta["arch_cfg"], meta["seed"])
    names = {n for n, _, _ in want}
    have = {n for n in tensors if n.startswith("model")}
    if names != have:
        raise SafeFormatError(f"{st}: model tensors differ from the network derived from code: missing "
                              f"{sorted(names - have)[:5]}, extra {sorted(have - names)[:5]}")
    for n, shape, dt in want:
        a = tensors[n]
        if a.shape != shape or a.dtype != dt:
            raise SafeFormatError(f"{st}: tensor {n!r} is {a.dtype}{a.shape}, the network expects {dt}{shape} "
                                  "(a float64 file needs jax_enable_x64; it is never down-cast)")
    shapes = {n: s for n, s, _ in want}
    norm = norm_from_tensors(tensors)
    _validate_norm(norm, meta["arch_cfg"]["n_k"], shapes["model.head_a.cddf.weight"][0],
                   shapes["model.head_a.dndx.weight"][0])
    model = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(tensors[n]) for n, _, _ in want])
    return model, meta, norm


# --------------------------------------------------------------------------------------------- bundle manifest
def _plain(name, what):
    if not isinstance(name, str) or not _PLAIN_NAME.match(name):
        raise SafeFormatError(f"{what} {name!r} is not a plain file name (no directories, no leading dot)")
    return name


def write_manifest(bundle_dir, member_prefixes, provenance):
    """``manifest.json`` tying every file of the bundle together (sha256, size), in member order. Members must sit
    directly in ``bundle_dir`` under plain names."""
    bd = os.path.abspath(bundle_dir)
    names, files = [], []
    for pre in member_prefixes:
        if os.path.dirname(os.path.abspath(pre)) != bd:
            raise SafeFormatError(f"member {pre!r} is not directly inside {bundle_dir}")
        m = _plain(os.path.basename(pre), "member")
        names.append(m)
        for ext in (".safetensors", ".json"):
            p = os.path.join(bd, m + ext)
            files.append({"path": m + ext, "sha256": sha256_file(p), "bytes": os.path.getsize(p)})
    if len(set(names)) != len(names):
        raise SafeFormatError("duplicate member names")
    man = {"format": FORMAT, "members": names, "files": files, "provenance": provenance}
    path = os.path.join(bd, MANIFEST)
    _write_exclusive(path, [json.dumps(man, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")])
    return path


def _manifest_entries(bundle_dir, expected_sha256=None, allow_extra=False):
    bd = os.path.abspath(bundle_dir)
    mp = os.path.join(bd, MANIFEST)
    if os.path.islink(mp):
        raise SafeFormatError(f"{mp} is a symlink")
    with open(mp, "rb") as f:
        raw = f.read()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SafeFormatError(f"{mp}: sha256 does not match the expected digest")
    man = _json_loads(raw, mp)
    if not isinstance(man, dict) or man.get("format") != FORMAT:
        raise SafeFormatError(f"manifest format {man.get('format') if isinstance(man, dict) else None!r}, "
                              f"expected {FORMAT}")
    members, files = man.get("members"), man.get("files")
    if not isinstance(members, list) or not members or not isinstance(files, list):
        raise SafeFormatError("manifest needs a non-empty members list and a files list")
    members = [_plain(m, "member") for m in members]
    if len(set(members)) != len(members):
        raise SafeFormatError("duplicate member names in the manifest")
    listed = {}
    for e in files:
        if not isinstance(e, dict) or set(e) != {"path", "sha256", "bytes"}:
            raise SafeFormatError(f"malformed manifest file entry {e!r:.80}")
        p = _plain(e["path"], "manifest path")
        if p in listed:
            raise SafeFormatError(f"manifest lists {p} twice")
        listed[p] = e
    want = {m + ext for m in members for ext in (".safetensors", ".json")}
    if set(listed) != want:
        raise SafeFormatError(f"manifest files {sorted(set(listed) ^ want)[:6]} do not match its members")
    for p, e in listed.items():
        fp = os.path.join(bd, p)
        if os.path.islink(fp) or not os.path.isfile(fp):
            raise SafeFormatError(f"bundle file {p} is missing, not a regular file, or a symlink")
        if os.path.getsize(fp) != e["bytes"] or sha256_file(fp) != e["sha256"]:
            raise SafeFormatError(f"bundle file {p} does not match the manifest")
    extra = sorted(set(os.listdir(bd)) - want - {MANIFEST})
    if extra and not allow_extra:
        raise SafeFormatError(f"bundle directory holds files the manifest does not list: {extra[:6]}")
    return [(os.path.join(bd, m), listed[m + ".safetensors"]["sha256"], listed[m + ".json"]["sha256"])
            for m in members]


def verify_manifest(bundle_dir, *, expected_sha256=None, allow_extra=False):
    """Check ``manifest.json`` against the directory: exactly the members' two files each, plain names, no symlinks,
    sizes and sha256, and (unless ``allow_extra``) no unlisted file. Returns the member prefixes in order."""
    return [p for p, _, _ in _manifest_entries(bundle_dir, expected_sha256, allow_extra)]


def load_bundle(bundle_dir, *, expected_manifest_sha256):
    """Verify the manifest, then load every member with the manifest's digests. Returns ``[(model, meta, norm)]``.
    ``expected_manifest_sha256`` is required: the release anchor. Passing ``None`` explicitly loads an unanchored
    bundle (development only)."""
    return [load_member(p, expected_sha256=s, expected_json_sha256=j)
            for p, s, j in _manifest_entries(bundle_dir, expected_manifest_sha256)]


# --------------------------------------------------------------------------------------------- HDF5 tables
def write_table_h5(path, arrays, attrs):
    """Numeric datasets plus str/int/float/bool scalar attributes in a NEW file (``format`` is reserved), written to a
    temporary name and published by hard link like every other file of the format."""
    import h5py
    if "format" in attrs:
        raise SafeFormatError("attribute 'format' is reserved")
    for k, v in attrs.items():
        if not isinstance(v, (str, int, float, bool)):
            raise SafeFormatError(f"attribute {k!r} must be a str, int, float or bool")
    for k, v in arrays.items():
        a = np.asarray(v)
        if not isinstance(k, str) or not _PLAIN_NAME.match(k) or a.dtype == object or a.dtype.kind not in "fiub":
            raise SafeFormatError(f"dataset {k!r}: dtype {a.dtype} is not numeric or the name is not plain")
    if os.path.lexists(path):
        raise SafeFormatError(f"{path} exists: never overwritten")
    fd, tmp = _new_tmp(path, ".h5")
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:                 # our own new temporary file
            f.attrs["format"] = FORMAT
            for k, v in attrs.items():
                f.attrs[k] = v
            for k, v in arrays.items():
                f.create_dataset(k, data=np.asarray(v))
        _finalize_tmp(tmp)
    except BaseException:
        os.remove(tmp)
        raise
    _publish(tmp, path)


def read_table_h5(path, required):
    """Read datasets ``required`` = {name: (dtype, shape)} exactly (``None`` in a shape matches any length).
    Refuses soft and external links, external or virtual storage, variable-length or object data, and attributes that
    are not str or numeric scalars."""
    import h5py
    out, attrs = {}, {}
    with h5py.File(path, "r") as f:
        for k in f.attrs:
            v = f.attrs[k]
            try:
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="strict")
                elif isinstance(v, str):
                    v.encode("utf-8", errors="strict")        # h5py surrogate-escapes invalid bytes
            except UnicodeError:
                raise SafeFormatError(f"{path}: attribute {k!r} is not UTF-8") from None
            if not isinstance(v, (str, np.integer, np.floating, np.bool_, int, float, bool)):
                raise SafeFormatError(f"{path}: attribute {k!r} is not a str or numeric scalar")
            attrs[k] = v
        if attrs.get("format") != FORMAT:
            raise SafeFormatError(f"{path}: format {attrs.get('format')!r}, expected {FORMAT}")
        for name, (dt, shape) in required.items():
            if not isinstance(name, str) or not _PLAIN_NAME.match(name):
                raise SafeFormatError(f"{path}: dataset name {name!r} must be a plain top-level name")
            link = f.get(name, getlink=True)
            if link is None:
                raise SafeFormatError(f"{path}: dataset {name!r} missing")
            if not isinstance(link, h5py.HardLink):
                raise SafeFormatError(f"{path}: {name!r} is a {type(link).__name__}, only hard links are read")
            d = f[name]
            if not isinstance(d, h5py.Dataset) or d.external or d.is_virtual:
                raise SafeFormatError(f"{path}: {name!r} is not a plain dataset")
            if (h5py.check_vlen_dtype(d.dtype) is not None or d.dtype.kind not in "fiub" or d.dtype != np.dtype(dt)
                    or d.ndim != len(shape) or any(s is not None and s != g for s, g in zip(shape, d.shape))):
                raise SafeFormatError(f"{path}: dataset {name!r} is {d.dtype}{d.shape}, expected {dt}{shape}")
            out[name] = d[...]
    return out, attrs
