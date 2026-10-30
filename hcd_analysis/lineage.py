"""Lineage record of a product, format ``hcd_priya.lineage/1``: what a product was made from, by which code and which
catalogue finder, and whether it may be used for a release.

A record is embedded in the product it describes (an ``.npz`` field ``lineage_json`` holding the canonical JSON; a JSON
field; in HDF5 a scalar string dataset, not an attribute, since compact HDF5 attributes are limited to 64 KiB), so the
product's own sha256 covers it. Each record lists the sha256 of every direct input,
so pinning the sha256 of a top product pins every ancestor back to the catalogues and the raw spectra.

Layout::

    {"format": "hcd_priya.lineage/1",
     "kind": "catalog", "kind_version": 2,          # what the product is, and its layout version
     "inputs": [{"role": "raw_spectra", "kind": "raw_spectra", "sha256": "...", "size": 123, "mtime_ns": 456, ...},
                {"role": "catalog", "kind": "catalog", "sha256": "...", "size": 789,
                 "finder_ids": ["..."], "release_eligible": true}],      # the last two: lineage-bearing inputs only
     "finder_ids": ["..."],                          # a catalogue: its finder; any other product: the union of its inputs'
     "settings": {...},                              # the product-determining settings
     "run_settings_sha256": "..." | null,            # sha256 of the output root's run_settings.json, if any
     "code": {"commit": "<40 hex>" | null, "dirty": false | true | null},
     "release_eligible": true | false, "ineligible_reasons": [...],
     "product": {...},                               # kind-specific facts (counts, payload digest), compared by the kind
     "validation": {...},                            # optional
     "created_utc": "..."}

**Release eligibility** (``eligibility``): a product is release-eligible only if the code commit is known and the tree
was clean (clean means the tracked files are those of the commit; untracked files are not looked at), every finder id
is in the release allow-list, every lineage-bearing input is itself eligible, and every other input is a raw file
(``RAW_INPUT_KINDS``) or one the record's ``validation`` block names as validated. It is computed when the record is
built and recomputed by every check, so a record that claims eligibility it does not have is refused. Historical
products carry ``release_eligible: false``. Eligibility is provenance only: it says which finder and which code made a
product, not that it was made with the release settings or from a whole file; release consumers check those
themselves (for a catalogue, ``catalog.load_validated_catalog`` with its expectations).

Standard library only; numpy is imported by the ``.npz`` helpers alone."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

FORMAT = "hcd_priya.lineage/1"
NPZ_KEY = "lineage_json"
MAX_RECORD_BYTES = 1 << 22

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_TOP_KEYS = {"format", "kind", "kind_version", "inputs", "finder_ids", "settings", "run_settings_sha256", "code",
             "release_eligible", "ineligible_reasons", "product", "validation", "created_utc"}
_OPTIONAL_TOP_KEYS = {"validation", "created_utc"}
_INPUT_KEYS = {"role", "kind", "sha256", "size", "mtime_ns", "finder_ids", "release_eligible", "info"}
# inputs that carry no lineage of their own and need none: simulation outputs read as they are (the raw spectra and
# the simulation's initial-conditions parameter file). Any other input must be lineage-bearing, or be named as
# validated in the record's validation block (``validation.validated_inputs``, the R1 carry's historical source).
RAW_INPUT_KINDS = frozenset({"raw_spectra", "sim_ics"})


class LineageError(ValueError):
    """A lineage record is missing, malformed, inconsistent with its product, or refused."""


# ------------------------------------------------------------------------------------------------------- hashing
def sha256_file(path, chunk=1 << 24):
    """sha256 (hex) of a file, streamed."""
    return file_identity(path, chunk)["sha256"]


def file_identity(path, chunk=1 << 24):
    """``{"sha256", "size", "mtime_ns"}`` of a file. The size and mtime are those of the descriptor that was hashed, so
    the three describe one file even if the name is replaced meanwhile."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        st = os.fstat(fh.fileno())
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def same_file(path, identity, rehash=False):
    """Whether ``path`` holds the bytes ``identity`` names (``sha256`` and ``size``). When the size and mtime_ns both
    match, the file is accepted without rereading it (the shortcut trusts that a file rewritten in place changes its
    mtime; ``rehash`` always hashes); otherwise it is hashed, so a copy with another mtime still matches by
    content."""
    st = os.stat(path)
    if int(st.st_size) != identity["size"]:
        return False
    if not rehash and identity.get("mtime_ns") is not None and int(st.st_mtime_ns) == identity["mtime_ns"]:
        return True
    return file_identity(path)["sha256"] == identity["sha256"]


def arrays_sha256(arrays):
    """Digest of named numpy arrays (name, dtype, shape and C-order bytes, in name order): the payload of an ``.npz``
    product without its embedded record."""
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[name]))
        if a.dtype.hasobject:
            raise LineageError(f"array {name!r} holds Python objects; it cannot be digested")
        h.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------- canonical JSON
def canonical_json(obj):
    """The record's canonical text: sorted keys, no whitespace, ASCII, no non-finite numbers."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonicalise(obj):
    """JSON round trip (tuples become lists, keys sorted), so a recorded value compares equal to what it was made from.
    Non-finite numbers raise ValueError."""
    return json.loads(canonical_json(obj))


def record_sha256(record):
    return hashlib.sha256(canonical_json(record).encode("ascii")).hexdigest()


def parse_record(text):
    """A record from its JSON text: strict JSON (no NaN or Infinity), bounded, then ``check_record``."""
    if not isinstance(text, str) or len(text) > MAX_RECORD_BYTES:
        raise LineageError("a lineage record must be JSON text of at most %d bytes" % MAX_RECORD_BYTES)

    def no_constant(c):
        raise ValueError(f"JSON constant {c}")
    try:
        record = json.loads(text, parse_constant=no_constant)
    except (ValueError, RecursionError) as e:
        raise LineageError(f"not a readable lineage record ({type(e).__name__}: {e})") from None
    return check_record(record)


# ------------------------------------------------------------------------------------------------- code identity
def code_identity(repo_dir=None):
    """``{"commit": <40 hex> | None, "dirty": bool | None}`` of the git checkout holding this package (or
    ``repo_dir``). None when git cannot tell, including when the package is not the top of a checkout (an installed
    copy). Tracked files only, without git's optional index lock, so asking never writes into the checkout."""
    root = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent.parent
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=str(root),
                                      stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        if Path(top).resolve() != root.resolve():
            return {"commit": None, "dirty": None}
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL,
                                         timeout=10).decode().strip()
        status = subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain",
                                          "--untracked-files=no"], cwd=str(root), stderr=subprocess.DEVNULL,
                                         timeout=30)
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit if _HEX40.match(commit) else None, "dirty": bool(status.strip())}


# --------------------------------------------------------------------------------------------------------- inputs
def input_entry(role, kind, identity, record=None, info=None):
    """One direct input. ``identity`` gives its ``sha256`` and ``size`` (and ``mtime_ns`` for a raw input, from
    ``file_identity``). ``record`` is the input's own lineage record when it is a lineage-bearing product: its finder
    ids and release eligibility are copied (a record a historical loader handed out is refused). Without ``record``
    the input is not lineage-bearing, and a product listing it is eligible only if its kind is in ``RAW_INPUT_KINDS``
    or the product's validation names it. ``info`` holds facts recorded but not compared (a path, a header)."""
    entry = {"role": str(role), "kind": str(kind), "sha256": identity["sha256"], "size": int(identity["size"])}
    if identity.get("mtime_ns") is not None:
        entry["mtime_ns"] = int(identity["mtime_ns"])
    if record is not None:
        if (record.get("product") or {}).get("historical_reason"):
            raise LineageError(f"input {role!r}: a record read through a historical loader is never listed as a "
                               "lineage-bearing input (list the file without its record; the product is then not "
                               "release-eligible)")
        entry["finder_ids"] = sorted(record["finder_ids"])
        entry["release_eligible"] = record["release_eligible"] is True
    if info is not None:
        entry["info"] = canonicalise(info)
    return entry


# ---------------------------------------------------------------------------------------------------- eligibility
def eligibility(record, allowed_finder_ids):
    """The reasons (a list; empty means eligible) why ``record`` is not release-eligible under the rule: a known commit
    and a clean tree (tracked files only), every finder id in ``allowed_finder_ids`` (at least one), every
    lineage-bearing input eligible with allowed finder ids, every other input raw (``RAW_INPUT_KINDS``) or named in
    ``validation.validated_inputs``, and no historical marker. Provenance only (see the module docstring)."""
    allowed = frozenset(allowed_finder_ids)
    why = []
    code = record.get("code") or {}
    if not (isinstance(code.get("commit"), str) and _HEX40.match(code["commit"])):
        why.append("the code commit is unknown")
    if code.get("dirty") is not False:
        why.append("the code tree was dirty" if code.get("dirty") is True else "the code tree state is unknown")
    fids = record.get("finder_ids") or []
    if not fids:
        why.append("no finder id")
    bad = sorted(f for f in fids if f not in allowed)
    if bad:
        why.append(f"finder ids not in the release allow-list: {bad}")
    validated = set((record.get("validation") or {}).get("validated_inputs") or [])
    for inp in record.get("inputs") or []:
        if "release_eligible" in inp or "finder_ids" in inp:
            if inp.get("release_eligible") is not True:
                why.append(f"input {inp.get('role')!r} is not release-eligible")
            ibad = sorted(f for f in inp.get("finder_ids") or [] if f not in allowed)
            if ibad or not inp.get("finder_ids"):
                why.append(f"input {inp.get('role')!r} has finder ids {inp.get('finder_ids')} outside the allow-list")
        elif inp.get("kind") not in RAW_INPUT_KINDS and inp.get("role") not in validated:
            why.append(f"input {inp.get('role')!r} of kind {inp.get('kind')!r} carries no lineage record (only "
                       f"{sorted(RAW_INPUT_KINDS)} inputs, or inputs the validation names, may)")
    if (record.get("product") or {}).get("historical_reason"):
        why.append("historical: " + str(record["product"]["historical_reason"]))
    return why


def build_record(kind, kind_version, *, inputs, finder_ids, settings, code, allowed_finder_ids,
                 run_settings_sha256=None, product=None, validation=None):
    """A complete, checked record. ``finder_ids`` is the product's own (a catalogue's finder) or None for the union of
    the lineage-bearing inputs' ids. Release eligibility is computed here (``eligibility``), never passed in."""
    if finder_ids is None:
        finder_ids = sorted({f for inp in inputs for f in inp.get("finder_ids", [])})
    record = {
        "format": FORMAT,
        "kind": str(kind),
        "kind_version": int(kind_version),
        "inputs": [dict(i) for i in inputs],
        "finder_ids": sorted(set(finder_ids)),
        "settings": settings,
        "run_settings_sha256": run_settings_sha256,
        "code": {"commit": code.get("commit"), "dirty": code.get("dirty")},
        "product": product or {},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if validation is not None:
        record["validation"] = validation
    record = canonicalise(record)
    why = eligibility(record, allowed_finder_ids)
    record["release_eligible"] = not why
    record["ineligible_reasons"] = why
    return check_record(record, allowed_finder_ids)


def check_record(record, allowed_finder_ids=None):
    """Refuse (LineageError) a record that is not a well-formed ``hcd_priya.lineage/1`` record. With
    ``allowed_finder_ids``, also refuse one whose ``release_eligible`` disagrees with the rule recomputed under that
    allow-list (a record never claims an eligibility it does not have). Returns the record."""
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise LineageError(f"not a {FORMAT} record")
    keys = set(record)
    missing = (_TOP_KEYS - _OPTIONAL_TOP_KEYS) - keys
    extra = keys - _TOP_KEYS
    if missing or extra:
        raise LineageError(f"lineage record keys: missing {sorted(missing)}, unknown {sorted(extra)}")
    if not (isinstance(record["kind"], str) and record["kind"]):
        raise LineageError("lineage record: kind must be a non-empty string")
    if not (isinstance(record["kind_version"], int) and not isinstance(record["kind_version"], bool)
            and record["kind_version"] >= 1):
        raise LineageError("lineage record: kind_version must be a positive integer")
    if not isinstance(record["inputs"], list):
        raise LineageError("lineage record: inputs must be a list")
    for inp in record["inputs"]:
        _check_input(inp)
    _check_finder_ids(record["finder_ids"], "finder_ids")
    if not isinstance(record["settings"], dict):
        raise LineageError("lineage record: settings must be an object")
    rs = record["run_settings_sha256"]
    if rs is not None and not (isinstance(rs, str) and _HEX64.match(rs)):
        raise LineageError("lineage record: run_settings_sha256 must be null or a 64-hex digest")
    code = record["code"]
    if not (isinstance(code, dict) and set(code) == {"commit", "dirty"}):
        raise LineageError("lineage record: code must be {commit, dirty}")
    if code["commit"] is not None and not (isinstance(code["commit"], str) and _HEX40.match(code["commit"])):
        raise LineageError("lineage record: code.commit must be null or a 40-hex commit")
    if code["dirty"] not in (True, False, None):
        raise LineageError("lineage record: code.dirty must be true, false or null")
    if not isinstance(record["release_eligible"], bool):
        raise LineageError("lineage record: release_eligible must be a boolean")
    if not (isinstance(record["ineligible_reasons"], list)
            and all(isinstance(r, str) for r in record["ineligible_reasons"])):
        raise LineageError("lineage record: ineligible_reasons must be a list of strings")
    if record["release_eligible"] == bool(record["ineligible_reasons"]):
        raise LineageError("lineage record: release_eligible disagrees with ineligible_reasons")
    if not isinstance(record["product"], dict):
        raise LineageError("lineage record: product must be an object")
    if "validation" in record and not isinstance(record["validation"], dict):
        raise LineageError("lineage record: validation must be an object")
    if allowed_finder_ids is not None and record["release_eligible"]:
        why = eligibility(record, allowed_finder_ids)
        if why:
            raise LineageError("lineage record claims release eligibility it does not have: " + "; ".join(why))
    return record


def _check_finder_ids(fids, where):
    if not (isinstance(fids, list) and all(isinstance(f, str) and f for f in fids) and fids == sorted(set(fids))):
        raise LineageError(f"lineage record: {where} must be a sorted list of distinct non-empty strings")


def _check_input(inp):
    if not isinstance(inp, dict) or set(inp) - _INPUT_KEYS:
        raise LineageError(f"lineage record: malformed input {inp!r}")
    for k in ("role", "kind"):
        if not (isinstance(inp.get(k), str) and inp[k]):
            raise LineageError(f"lineage record: input {k} must be a non-empty string")
    if not (isinstance(inp.get("sha256"), str) and _HEX64.match(inp["sha256"])):
        raise LineageError(f"lineage record: input {inp.get('role')!r} has no 64-hex sha256")
    for k in ("size", "mtime_ns"):
        if k in inp and not (isinstance(inp[k], int) and not isinstance(inp[k], bool) and inp[k] >= 0):
            raise LineageError(f"lineage record: input {inp.get('role')!r} {k} must be a non-negative integer")
    if "size" not in inp:
        raise LineageError(f"lineage record: input {inp.get('role')!r} has no size")
    if ("finder_ids" in inp) != ("release_eligible" in inp):
        raise LineageError(f"lineage record: input {inp.get('role')!r} must carry both finder_ids and "
                           "release_eligible, or neither")
    if "finder_ids" in inp:
        _check_finder_ids(inp["finder_ids"], f"input {inp['role']!r} finder_ids")
        if not isinstance(inp["release_eligible"], bool):
            raise LineageError(f"lineage record: input {inp['role']!r} release_eligible must be a boolean")


def find_input(record, role):
    """The single input of ``record`` with this role (LineageError if there is not exactly one)."""
    hits = [i for i in record["inputs"] if i["role"] == role]
    if len(hits) != 1:
        raise LineageError(f"lineage record ({record['kind']}): {len(hits)} inputs with role {role!r}, expected 1")
    return hits[0]


def verify_input(record, role, path, rehash=False):
    """Refuse (LineageError) unless the file at ``path`` holds the bytes the record names for the input ``role``
    (``same_file``)."""
    inp = find_input(record, role)
    if not same_file(path, inp, rehash=rehash):
        raise LineageError(f"{path}: not the {role} input this {record['kind']} was made from (sha256 differs)")
    return inp


# ------------------------------------------------------------------------------------------------------------ npz
def write_npz(path, arrays, record):
    """Write ``arrays`` and ``record`` (field ``lineage_json``) to ``path``: to a temporary file in the same directory,
    then renamed into place, so a reader never sees a partial file. The record's ``product.payload_sha256`` must be
    the digest of ``arrays``."""
    import numpy as np
    check_record(record)
    if NPZ_KEY in arrays:
        raise LineageError(f"{NPZ_KEY} is reserved for the lineage record")
    if record["product"].get("payload_sha256") != arrays_sha256(arrays):
        raise LineageError("the record's payload_sha256 is not the digest of the arrays being written")
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays, **{NPZ_KEY: np.array(canonical_json(record))})
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_npz(path, require_record=True):
    """``(arrays, record)`` of an ``.npz`` product, read without pickle. The record is checked for form, and when it
    names ``product.payload_sha256`` the arrays must match it. Without a record: LineageError, or ``(arrays, None)``
    when ``require_record`` is false."""
    import numpy as np
    with np.load(str(path), allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    raw = arrays.pop(NPZ_KEY, None)
    if raw is None:
        if require_record:
            raise LineageError(f"{path}: no lineage record ({NPZ_KEY})")
        return arrays, None
    if raw.ndim != 0 or raw.dtype.kind != "U":
        raise LineageError(f"{path}: {NPZ_KEY} is not a 0-d string")
    record = parse_record(str(raw[()]))
    payload = record["product"].get("payload_sha256")
    if payload is not None and payload != arrays_sha256(arrays):
        raise LineageError(f"{path}: the arrays do not match the record's payload_sha256 (edited after it was made)")
    return arrays, record
