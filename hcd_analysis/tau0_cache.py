"""The tau0 emulator training cache file, format 3.4: its datasets, its lineage record, its readers and its writer.

A v3.4 cache carries a ``hcd_priya.lineage/1`` record (kind ``tau0_cache``) as the scalar string dataset
``lineage_json`` (a dataset, not an attribute: a record over about a thousand groups exceeds the 64 KiB limit of a
compact HDF5 attribute).
The record lists every direct input by sha256:
  * per (simulation, snapshot) group: the snapshot record (``snapshot:<sim>:<snap>``, meta.json), the catalogue
    (``catalog:...``) and its CDDF (``cddf:...``), each with its finder ids and release eligibility, and the raw
    spectra file (``raw_spectra:...``, sha256, size and mtime as the catalogue build hashed it);
  * per simulation: its SimulationICs.json (``sim_ics:<sim>``);
  * any cache it was merged, spliced or carried from.
``product.payload_sha256`` digests every other dataset and every attribute, so a dataset edited after the record was
written is refused. ``product.groups`` lists the groups in file order with the origin of their rows: ``built`` (by the
cache builder from raw spectra) or ``carried_validated`` (copied from a historical cache by the carry tool,
scripts/carry_validated_cache.py, after its per-group validation; ``product.carried`` names the source).

Readers:
  * ``read_cache(path)``: the release reader. Format 3.4 only, a well-formed record of kind ``tau0_cache``, the payload
    digest, and release eligibility; anything else is refused.
  * ``read_cache_historical(path, reason=...)``: any 3.3 or 3.4 file, no lineage checks; the only reader of a cache
    written before caches carried lineage. The reason is logged.
The writer (``write_cache``) writes to a temporary file in the target directory, digests what it wrote, attaches the
record and renames the file into place; it never overwrites an existing file.

numpy and h5py only (no jax, no fake_spectra), so the builder, the carry tool and the training loader share it."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from . import lineage as L
from .catalog import RELEASE_FINDER_IDS, RELEASE_FINDER_SETTINGS, load_catalog_historical
from .snapshot_record import verify_snapshot_dir

logger = logging.getLogger(__name__)

CACHE_VERSION = "3.4"
HISTORICAL_VERSIONS = ("3.3", "3.4")
KIND = "tau0_cache"
KIND_VERSION = 1
LINEAGE_KEY = "lineage_json"              # the record's dataset

# dataset layout (unchanged from 3.3; 3.4 adds the lineage record only)
ROW_FLOAT_KEYS = ("alpha_slope", "target_F", "scale", "z_meta", "z_grid", "dv_kms")
ROW_INT_KEYS = ("snap", "alpha_idx", "nbins_native", "snap_group_idx")
ROW_P1D_KEYS = ("kfkms", "P_tier_p")                     # 2-D (n_rows, n_k)
ROW_TIERC_KEYS = ("P_tier_c", "P_tier_c_filtered")       # 3-D (n_rows, N_TIER_C_BINS, n_k)
ROW_COUNT_KEYS = ("tier_c_counts",)                      # 2-D (n_rows, N_TIER_C_BINS)
ROW_MEANF_KEYS = ("mean_F_by_bin",)                      # 2-D (n_rows, N_TIER_C_BINS)
SNAP_FLOAT_KEYS = ("total_path_dX", "dNdX_LLS", "dNdX_subDLA", "dNdX_DLA")
SNAP_2D_KEYS = ("f_nhi", "n_absorbers")
STATIC_KEYS = ("tier_c_labels", "tier_c_nhi_edges", "param_names", "log_nhi_centres", "log_nhi_edges")
ROW_KEYS = (("sim_name", "params") + ROW_P1D_KEYS + ROW_TIERC_KEYS + ROW_COUNT_KEYS + ROW_MEANF_KEYS
            + ROW_FLOAT_KEYS + ROW_INT_KEYS)
SNAP_KEYS = ("snap_sim_name", "snap_snap") + tuple("snap_" + k for k in SNAP_FLOAT_KEYS + SNAP_2D_KEYS)
ALL_KEYS = frozenset(ROW_KEYS + SNAP_KEYS + STATIC_KEYS)
STR_KEYS = frozenset({"sim_name", "snap_sim_name", "tier_c_labels", "param_names"})

ROWS_BUILT = "built"
ROWS_CARRIED = "carried_validated"
# the row fields the carry tool copies from a historical cache (functions of the raw optical depth, the sightline
# labels and the mean-flux rung only); every other field of a carried group is rebuilt
CARRIED_FIELDS = ("P_tier_p", "P_tier_c", "P_tier_c_filtered", "mean_F_by_bin", "scale", "target_F", "kfkms")
GROUP_INPUT_KINDS = ("snapshot", "catalog", "cddf", "raw_spectra")


class CacheLineageError(ValueError):
    """A training cache, or a snapshot offered as its input, is refused: wrong format, no or a malformed lineage record,
    a payload that differs from the record, not release-eligible, or inputs that are not what the record names."""


# --------------------------------------------------------------------------------------------- coarse classes
# The 15 fine Tier-C classes collapse to 4 coarse ones (clean, LLS, subDLA, DLA); the training loader
# (hcd_analysis.emulator.data) and the carry tool's DLA-core check use these.
COARSE_SLICES = (slice(0, 1), slice(1, 8), slice(8, 13), slice(13, 15))
COARSE_NAMES = ("clean", "LLS", "subDLA", "DLA")


def _collapse_counts(counts15):
    return np.stack([counts15[:, s].sum(1) for s in COARSE_SLICES], axis=1)


def _collapse_p1d(p15, counts15):
    """Count-weighted collapse of fine-bin P1D to coarse classes (R,15,K)->(R,4,K).

    NaN-safe over fine bins. An *empty* coarse class (zero counts) collapses to
    0.0, not NaN: the production cache stores finite zeros for empty classes
    (see ``priya_p1d._per_class_p1d_at_scale``), and the downstream structural
    sum ``P_tier_p = Σ_c w_c·P_filt`` weights an empty class by ``w_c=0``; a
    NaN there would poison the whole total via ``0·NaN=NaN``. Only a bin that is
    NaN across *all* fine sub-bins (genuinely above native Nyquist) stays NaN so
    it can be masked.

    Corner case: when a single k mixes finite and NaN fine sub-bins within one
    coarse class, NaN sub-bins contribute 0 while their weight stays in the
    denominator, slightly under-weighting the surviving finite sub-bins. This is
    a documented, accepted approximation (no behaviour change)."""
    R, _, K = p15.shape
    out = np.full((R, 4, K), np.nan)
    for ci, s in enumerate(COARSE_SLICES):
        c = counts15[:, s].astype(float)
        w = c / np.where(c.sum(1, keepdims=True) == 0, 1.0, c.sum(1, keepdims=True))
        seg = p15[:, s, :]
        contrib = np.where(np.isfinite(seg), seg, 0.0) * w[:, :, None]
        summ = contrib.sum(1)
        # Fires only for the Nyquist-NaN case (ALL fine sub-bins NaN), NOT for
        # zero-count empties (those have finite zeros and correctly stay 0.0).
        allnan = (~np.isfinite(seg)).all(1)
        summ[allnan] = np.nan
        out[:, ci, :] = summ
    return out


def dla_core_rows(P_tier_c, P_tier_c_filtered, tier_c_counts):
    """Per-row DLA core, the loader's ``delta[:, 2]``: the coarse DLA class of the unfiltered minus the filtered
    per-class P1D, collapsed with the rows' own counts."""
    return (_collapse_p1d(P_tier_c, tier_c_counts) - _collapse_p1d(P_tier_c_filtered, tier_c_counts))[:, 3, :]


# ------------------------------------------------------------------------------------------------ the payload
def _digest_dataset(h, name, arr):
    arr = np.asarray(arr)
    if name in STR_KEYS or arr.dtype.kind in "OSU":
        h.update(json.dumps([name, "str", list(arr.shape)]).encode("utf-8"))
        h.update(json.dumps([v.decode() if isinstance(v, bytes) else str(v) for v in arr.ravel().tolist()]).encode())
    else:
        a = np.ascontiguousarray(arr)
        h.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode("utf-8"))
        h.update(a.data if a.size else b"")


def _attr_value(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [_attr_value(x) for x in v.tolist()] if v.dtype.kind in "OSU" else v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _digest_attrs(h, attrs):
    h.update(L.canonical_json({"attrs": {k: _attr_value(v) for k, v in attrs.items()}}).encode())


def payload_sha256(datasets, attrs):
    """Digest of a cache's datasets other than the lineage record (name, dtype, shape and bytes, or the strings, in
    name order) and of its attributes."""
    h = hashlib.sha256()
    for name in sorted(k for k in datasets if k != LINEAGE_KEY):
        _digest_dataset(h, name, datasets[name])
    _digest_attrs(h, attrs)
    return h.hexdigest()


def payload_sha256_of_file(path):
    """``payload_sha256`` of a cache file, reading one dataset at a time."""
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for name in sorted(k for k in f.keys() if k != LINEAGE_KEY):
            _digest_dataset(h, name, f[name].asstr()[...] if name in STR_KEYS else f[name][...])
        _digest_attrs(h, dict(f.attrs))
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------ readers
def read_datasets(path):
    """``(datasets, attrs, record_text)`` of a cache file: every dataset but the record (strings as str), every
    attribute, and the record's JSON text (None if the file has none). The key set must be the tau0 cache layout."""
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if keys - {LINEAGE_KEY} != ALL_KEYS:
            raise CacheLineageError(f"{path}: key set differs from the tau0 cache layout: missing "
                                    f"{sorted(ALL_KEYS - keys)}, unknown {sorted(keys - ALL_KEYS - {LINEAGE_KEY})}")
        d = {k: (f[k].asstr()[...] if k in STR_KEYS else f[k][...]) for k in sorted(ALL_KEYS)}
        attrs = {k: f.attrs[k] for k in f.attrs}
        text = None
        if LINEAGE_KEY in keys:
            ds = f[LINEAGE_KEY]
            if ds.shape != () or ds.dtype.kind != "O":
                raise CacheLineageError(f"{path}: {LINEAGE_KEY} is not a scalar string dataset")
            text = ds.asstr()[()]
    return d, attrs, text


def read_cache(path):
    """The release reader: ``(datasets, attrs, record)``. Refused (CacheLineageError) unless the file is format 3.4
    with a well-formed ``tau0_cache`` record, its datasets and attributes match the record's payload digest, its groups
    match the record, and it is release-eligible. Carried rows must also be promoted (``product.carried.promoted``:
    the fresh-row comparison passed; scripts/promote_carried_cache.py)."""
    return _read_release(path, staged_ok=False)


def read_cache_staged(path):
    """``read_cache`` without the promotion requirement: reads a carried cache the carry tool wrote before the fresh-row
    comparison has passed. Only the fresh-row comparison (scripts/fresh_row_sample.py) and the promotion step read a
    cache this way; no training, product or model reads a staged cache."""
    return _read_release(path, staged_ok=True)


def _read_release(path, *, staged_ok):
    def refuse(why):
        raise CacheLineageError(f"{path}: training cache refused ({why})")
    d, attrs, text = read_datasets(path)
    ver = str(_attr_value(attrs.get("cache_version")))
    if ver != CACHE_VERSION:
        refuse(f"cache version {ver!r}; the release reader reads {CACHE_VERSION} only. A cache written before caches "
               "carried lineage is historical: read it with read_cache_historical(path, reason=...) or "
               "hcd_analysis.emulator.data.load_cache_historical, or rebuild it")
    if text is None:
        refuse(f"no lineage record ({LINEAGE_KEY})")
    try:
        rec = L.check_record(L.parse_record(text), RELEASE_FINDER_IDS)
    except L.LineageError as e:
        refuse(str(e))
    if rec["kind"] != KIND or rec["kind_version"] != KIND_VERSION:
        refuse(f"a {rec['kind']} record of version {rec['kind_version']}, not {KIND} {KIND_VERSION}")
    if rec["product"].get("payload_sha256") != payload_sha256(d, attrs):
        refuse("its datasets or attributes differ from the record's payload digest (edited after it was written)")
    if not rec["release_eligible"]:
        refuse("not release-eligible: " + "; ".join(rec["ineligible_reasons"]))
    check_groups(rec, d, refuse, staged_ok=staged_ok)
    return d, attrs, rec


def check_groups(rec, d, refuse, *, staged_ok=False):
    """The record's groups are the file's snapshot blocks, in order; each has its snapshot, catalogue and CDDF inputs
    (each of its own kind, lineage-bearing and release-eligible) and its raw input (of kind raw_spectra); carried
    groups name the carry's source and the carried fields and, unless ``staged_ok``, the promotion (the fresh-row
    comparison that passed, by sha256). ``validation.validated_inputs`` (the inputs that may carry no lineage of their
    own) is allowed only with carried groups, and must then name exactly the carry's source: one ``tau0_cache`` input
    whose sha256 and size are the carry block's."""
    groups = rec["product"].get("groups")
    ids = list(zip([str(s) for s in d["snap_sim_name"]], [int(s) for s in d["snap_snap"]]))
    if not isinstance(groups, list) or [(g.get("sim"), g.get("snap")) for g in groups] != ids:
        refuse("the record's groups are not the file's snapshot blocks")
    if rec["product"].get("n_rows") != int(np.asarray(d["sim_name"]).shape[0]) or rec["product"].get("n_snaps") \
            != len(ids):
        refuse("the record's row or group count is not the file's")
    by_role = {}
    for i in rec["inputs"]:
        by_role.setdefault(i["role"], []).append(i)
    for g in groups:
        if g.get("rows") not in (ROWS_BUILT, ROWS_CARRIED):
            refuse(f"group {g.get('sim')} {g.get('snap')}: unknown row origin {g.get('rows')!r}")
        missing = [k for k in GROUP_INPUT_KINDS if group_role(k, g["sim"], g["snap"]) not in by_role]
        if missing:
            refuse(f"group {g['sim']} {g['snap']}: no {missing} input in the record")
        for k in GROUP_INPUT_KINDS:
            role = group_role(k, g["sim"], g["snap"])
            hits = by_role[role]
            if len(hits) != 1 or hits[0]["kind"] != k:
                refuse(f"group {g['sim']} {g['snap']}: the input {role} is not one input of kind {k}")
            if k != "raw_spectra" and not (hits[0].get("finder_ids") and hits[0].get("release_eligible") is True):
                refuse(f"group {g['sim']} {g['snap']}: the input {role} is not a release-eligible product with "
                       "finder ids")
    validation = rec.get("validation") or {}
    validated = list(validation.get("validated_inputs") or [])
    carried_any = any(g["rows"] == ROWS_CARRIED for g in groups)
    if validated and not carried_any:
        refuse("validated inputs without carried rows (only the carry's source may be named as validated)")
    if carried_any:
        car = rec["product"].get("carried") or {}
        if list(car.get("fields") or []) != list(CARRIED_FIELDS) or not isinstance(car.get("source_sha256"), str):
            refuse("carried rows without the carry's source and field list")
        if not validation.get("all_passed") is True:
            refuse("carried rows without a passed validation")
        src = by_role.get(validated[0], []) if len(validated) == 1 else []
        if not (len(src) == 1 and src[0]["kind"] == KIND and src[0]["sha256"] == car.get("source_sha256")
                and src[0]["size"] == car.get("source_size")):
            refuse("carried rows: the validated input must be exactly the carry's source (one tau0_cache input "
                   "whose sha256 and size are the carry block's)")
        prom = car.get("promoted")
        if not staged_ok and not (isinstance(prom, dict) and prom.get("passed") is True
                                  and all(isinstance(prom.get(k), str) and len(prom[k]) == 64
                                          for k in ("comparison_sha256", "staged_sha256"))):
            refuse("carried rows not yet promoted: a staged carry is usable only after the fresh-row comparison "
                   "passed (scripts/promote_carried_cache.py)")


def read_cache_historical(path, *, reason):
    """``(datasets, attrs, record or None)`` of a 3.3 or 3.4 cache, with no lineage check: the only reader of a cache
    written before caches carried lineage. ``reason`` (non-empty) is logged; nothing read this way may enter a
    release product except through the carry tool's validation."""
    if not (isinstance(reason, str) and reason.strip()):
        raise ValueError("read_cache_historical needs a non-empty reason")
    d, attrs, text = read_datasets(path)
    ver = str(_attr_value(attrs.get("cache_version")))
    if ver not in HISTORICAL_VERSIONS:
        raise CacheLineageError(f"{path}: cache version {ver!r} is not one of {HISTORICAL_VERSIONS}")
    rec = None
    if text is not None:
        try:
            rec = L.parse_record(text)
        except L.LineageError:
            rec = None
    logger.warning("historical read of the cache %s (%s): not release-eligible", path, reason.strip())
    return d, attrs, rec


def read_groups_historical(path, *, reason):
    """The group list of a 3.3 or 3.4 cache without reading its row arrays: ``(groups, attrs)`` with groups
    ``[(sim, snap, z_grid, n_rows)]`` in file order. ``reason`` is logged."""
    if not (isinstance(reason, str) and reason.strip()):
        raise ValueError("read_groups_historical needs a non-empty reason")
    with h5py.File(path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
        ver = str(_attr_value(attrs.get("cache_version")))
        if ver not in HISTORICAL_VERSIONS:
            raise CacheLineageError(f"{path}: cache version {ver!r} is not one of {HISTORICAL_VERSIONS}")
        sims, snaps = f["snap_sim_name"].asstr()[...], f["snap_snap"][...]
        gi, zg = f["snap_group_idx"][...], f["z_grid"][...]
    groups = []
    for b, (s, n) in enumerate(zip(sims, snaps)):
        rows = np.flatnonzero(gi == b)
        groups.append((str(s), int(n), float(zg[rows[0]]) if rows.size else float("nan"), int(rows.size)))
    logger.warning("historical read of the group list of %s (%s)", path, reason.strip())
    return groups, attrs


# ------------------------------------------------------------------------------------------------------- writer
def group_role(kind, sim, snap):
    return f"{kind}:{sim}:{int(snap):03d}"


def cache_settings(*, fidelity, n_k, alpha_refine, alpha_range, n_skewers=None):
    """The product-determining settings a cache record holds (equal for caches that may be merged)."""
    return L.canonicalise({"cache_version": CACHE_VERSION, "fidelity": str(fidelity), "n_k": int(n_k),
                           "alpha_refine": int(alpha_refine), "alpha_range": [float(a) for a in alpha_range],
                           "n_skewers": None if n_skewers is None else int(n_skewers), "tau_thresh": 1.0e6,
                           "tier_c_recipe": "uniform"})


def combine_inputs(*entry_lists):
    """One input list from several: an identical entry listed twice is kept once; one role naming two different
    inputs is refused."""
    out: Dict[str, dict] = {}
    for entries in entry_lists:
        for e in entries:
            e = L.canonicalise(e)
            if e["role"] in out and out[e["role"]] != e:
                raise CacheLineageError(f"input role {e['role']!r} names two different inputs")
            out.setdefault(e["role"], e)
    return list(out.values())


def publish_once(tmp, path):
    """Publish the finished file ``tmp`` at ``path`` write-once and remove the name ``tmp``. A hard link fails
    atomically when ``path`` exists, so a file another writer published there after the caller's existence check
    (another run with the same output) is never replaced: that is a FileExistsError, and ``path`` is left as it is.
    An EEXIST is accepted only when ``path`` is ``tmp``'s own inode and not a symlink (an NFS retransmit can report
    EEXIST for the link just made). The link never follows a symlink. The directory is fsynced where the platform
    allows it."""
    tmp, path = Path(tmp), Path(path)
    try:
        st_t = os.lstat(tmp)
        if not stat.S_ISREG(st_t.st_mode):
            raise CacheLineageError(f"{tmp} is not a regular file; nothing published at {path}")
        try:
            os.link(tmp, path, follow_symlinks=False)
        except FileExistsError:
            st_p = os.lstat(path)
            if stat.S_ISLNK(st_p.st_mode) or (st_p.st_dev, st_p.st_ino) != (st_t.st_dev, st_t.st_ino):
                raise FileExistsError(f"{path} exists (published by another writer after this run's check); a cache "
                                      f"is never overwritten: nothing of this run is published") from None
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def write_cache(path, datasets, attrs, *, inputs, settings, groups, carried=None, validation=None,
                historical_reason=None, selection=None):
    """Write a v3.4 cache and its lineage record; return the record. ``datasets`` is the full tau0 layout, ``attrs``
    the descriptive attributes (``cache_version`` is set here). Eligibility is computed from ``inputs`` and the code
    identity (lineage.build_record); ``historical_reason`` makes the product ineligible. The file is written under a
    temporary name in the target directory, digested as written, given its record and published write-once
    (``publish_once``: a hard link, so an existing ``path`` is refused, before the write and at the publish, and
    never replaced)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a cache is never overwritten")
    if set(datasets) != ALL_KEYS:
        raise CacheLineageError(f"datasets differ from the tau0 cache layout: missing {sorted(ALL_KEYS - set(datasets))}"
                                f", unknown {sorted(set(datasets) - ALL_KEYS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        with h5py.File(tmp, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["cache_version"] = CACHE_VERSION
            for k in sorted(datasets):
                v = datasets[k]
                if k in STR_KEYS:
                    f.create_dataset(k, data=np.array([str(x) for x in np.asarray(v).ravel()], dtype=object),
                                     dtype=h5py.string_dtype())
                else:
                    f.create_dataset(k, data=np.asarray(v))
        product = {"payload_sha256": payload_sha256_of_file(tmp), "n_rows": int(np.asarray(datasets["sim_name"]).size),
                   "n_snaps": int(np.asarray(datasets["snap_snap"]).size), "groups": groups}
        if carried is not None:
            product["carried"] = carried
        if selection is not None:
            product["selection"] = selection
        if historical_reason:
            product["historical_reason"] = str(historical_reason)
        rec = L.build_record(KIND, KIND_VERSION, inputs=inputs, finder_ids=None, settings=settings,
                             code=L.code_identity(), allowed_finder_ids=RELEASE_FINDER_IDS, product=product,
                             validation=validation)
        text = L.canonical_json(rec)
        if len(text) > L.MAX_RECORD_BYTES:
            raise CacheLineageError("the lineage record exceeds the readable size")
        with h5py.File(tmp, "a") as f:
            f.create_dataset(LINEAGE_KEY, data=text, dtype=h5py.string_dtype())
        ffd = os.open(str(tmp), os.O_RDONLY)                       # the file's bytes on disk before the publish
        try:
            os.fsync(ffd)
        finally:
            os.close(ffd)
        publish_once(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return rec


# ------------------------------------------------------------------------------------------- snapshot inputs
def load_snapshot_inputs(snap_dir, raw_path, *, sim=None, snap=None, rehash_raw=False):
    """A (simulation, snapshot)'s products as the cache builder and the carry tool use them: ``(verified, entries)``
    with ``verified`` from ``snapshot_record.verify_snapshot_dir`` (catalogue-only or full mode; the release finder
    settings ``RELEASE_FINDER_SETTINGS``; the raw file ``raw_path``, re-hashed with ``rehash_raw``; the file's skewer
    count; release eligibility of the catalogue, the cddf and the snapshot record; the copied flags; ``sim`` and
    ``snap`` when given) and ``entries`` the cache record's input entries for the group: the snapshot record
    (meta.json), the catalogue, the CDDF and the raw spectra file."""
    v = verify_snapshot_dir(snap_dir, settings=RELEASE_FINDER_SETTINGS, raw_path=raw_path, rehash=rehash_raw,
                            require_release_eligible=True, sim=sim, snap=snap)
    s, n = v.catalog.sim_name, int(v.catalog.snap)
    raw = L.find_input(v.catalog.lineage, "raw_spectra")
    entries = [
        L.input_entry(group_role("snapshot", s, n), "snapshot", v.meta_identity, record=v.record),
        L.input_entry(group_role("catalog", s, n), "catalog", v.products["catalog.npz"], record=v.catalog.lineage),
        L.input_entry(group_role("cddf", s, n), "cddf", v.products["cddf.npz"], record=v.cddf_record),
        L.input_entry(group_role("raw_spectra", s, n), "raw_spectra", raw, info={"path": str(raw_path)}),
    ]
    return v, entries


class HistoricalSnapshot:
    """A snapshot's products read as they are (``load_snapshot_inputs_historical``): ``meta``, ``cddf``, ``catalog``."""

    def __init__(self, meta, cddf, catalog):
        self.meta, self.cddf, self.catalog = meta, cddf, catalog


def load_snapshot_inputs_historical(snap_dir, *, reason):
    """A snapshot's products read as they are, for a historical reproduction: meta.json, cddf.npz (no pickle) and the
    catalogue through ``load_catalog_historical``. The entries carry the files' sha256 only (no finder ids, no raw
    file), so any product built from them is not release-eligible."""
    snap_dir = Path(snap_dir)
    meta_bytes = (snap_dir / "meta.json").read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    cddf, _ = L.read_npz(snap_dir / "cddf.npz", require_record=False)
    catalog = load_catalog_historical(snap_dir / "catalog.npz", reason=reason)
    s, n = str(meta["sim_name"]), int(meta["snap"])

    def ident(name):
        return {k: v for k, v in L.file_identity(snap_dir / name).items() if k != "mtime_ns"}
    entries = [L.input_entry(group_role("snapshot", s, n), "snapshot",
                             {"sha256": hashlib.sha256(meta_bytes).hexdigest(), "size": len(meta_bytes)}),
               L.input_entry(group_role("catalog", s, n), "catalog", ident("catalog.npz")),
               L.input_entry(group_role("cddf", s, n), "cddf", ident("cddf.npz"))]
    return HistoricalSnapshot(meta, cddf, catalog), entries


def sim_ics_entry(raw_path, sim):
    """The input entry of a simulation's SimulationICs.json (two levels above its SPECTRA directory), from which the
    builder reads the parameter vector."""
    p = Path(raw_path).resolve().parents[2] / "SimulationICs.json"
    ident = L.file_identity(p)
    return L.input_entry(f"sim_ics:{sim}", "sim_ics", {"sha256": ident["sha256"], "size": ident["size"]},
                         info={"path": str(p)})
