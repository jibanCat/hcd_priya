"""Run-settings record: the effective settings of a run, written once into its output directory.

An output directory holds the results of ONE set of settings. The first run writes ``run_settings.json`` there; every
later run into the same directory must carry identical settings, or it is refused (``RunSettingsMismatch``) before it
reads or writes anything else. A rerun therefore never silently mixes results made with different settings (for
example another ``absorber.tau_threshold``) with existing ones. The record is published atomically (a complete
temporary file linked into place, never replacing an existing record), so two processes starting together agree on
one record. Fields under ``info`` (code commit, creation time) are recorded for traceability and never compared.

Layout of the record (format ``hcd_priya.run_settings/1``)::

    {"format": "hcd_priya.run_settings/1", "settings": {...compared...}, "info": {...not compared...}}
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

RUN_SETTINGS_FILE = "run_settings.json"
FORMAT = "hcd_priya.run_settings/1"
MAX_RECORD_BYTES = 1 << 20


class RunSettingsMismatch(RuntimeError):
    """The settings of a run differ from those recorded in its output directory, or the record is unreadable."""


def _canonical(obj):
    """JSON round trip (tuples become lists, keys sorted), so a recorded value compares equal to the value it was made
    from. Non-finite numbers raise ValueError."""
    return json.loads(json.dumps(obj, sort_keys=True, allow_nan=False))


def _differences(old, new, prefix=""):
    """Dotted keys at which two canonical settings trees differ (a key present on one side only counts)."""
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return [] if old == new else [prefix or "<settings>"]
    out = []
    for k in sorted(set(old) | set(new)):
        key = f"{prefix}.{k}" if prefix else k
        if k not in old or k not in new:
            out.append(key)
        else:
            out.extend(_differences(old[k], new[k], key))
    return out


def _lookup(tree, dotted):
    for part in dotted.split("."):
        if not isinstance(tree, dict) or part not in tree:
            return "<absent>"
        tree = tree[part]
    return tree


def _read_record(path):
    """The record's JSON, read without following a symlink and bounded in size; RunSettingsMismatch otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as e:
        raise RunSettingsMismatch(f"{path}: the run-settings record cannot be read ({e.strerror})") from None
    with os.fdopen(fd, "rb") as fh:
        raw = fh.read(MAX_RECORD_BYTES + 1)
    if len(raw) > MAX_RECORD_BYTES:
        raise RunSettingsMismatch(f"{path}: larger than {MAX_RECORD_BYTES} bytes; not a run-settings record")

    def no_constant(c):
        raise ValueError(f"JSON constant {c}")
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=no_constant)
    except ValueError as e:
        raise RunSettingsMismatch(f"{path}: not a readable run-settings record ({e})") from None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT or not isinstance(doc.get("settings"), dict):
        raise RunSettingsMismatch(f"{path}: not a {FORMAT} record")
    return doc


def _publish(out_dir, path, doc):
    """Write ``doc`` to a temporary file in ``out_dir`` and link it to ``path``; an existing record is never replaced
    (the link fails), in which case the caller verifies against it."""
    blob = (json.dumps(doc, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
    tmp = out_dir / f".{RUN_SETTINGS_FILE}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass                                    # another process recorded first: verified by the caller
    finally:
        os.unlink(tmp)


def record_or_verify(out_dir, settings, info=None):
    """Record ``settings`` in ``out_dir/run_settings.json`` if no record exists, else verify them against it.

    ``info`` is an optional callable returning a dict of fields recorded but never compared; it is called only when
    a record is written. Returns the record's path. Raises ``RunSettingsMismatch`` when the recorded settings differ
    (the message names every differing key with both values) or the record is unreadable, and ``ValueError`` for
    non-finite settings (before anything is written)."""
    out_dir = Path(out_dir)
    path = out_dir / RUN_SETTINGS_FILE
    want = _canonical(settings)
    if not os.path.lexists(path):
        out_dir.mkdir(parents=True, exist_ok=True)
        extra = _canonical(info() if info is not None else {})
        extra.setdefault("created_utc", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        _publish(out_dir, path, {"format": FORMAT, "settings": want, "info": extra})
    recorded = _read_record(path)["settings"]
    diff = _differences(recorded, want)
    if diff:
        detail = "; ".join(f"{k}: recorded {_lookup(recorded, k)!r}, this run {_lookup(want, k)!r}" for k in diff[:12])
        more = f" (and {len(diff) - 12} more)" if len(diff) > 12 else ""
        raise RunSettingsMismatch(
            f"{out_dir} holds results made with different settings: {detail}{more}. Nothing was read or written; "
            "use a new output directory for these settings.")
    return path
