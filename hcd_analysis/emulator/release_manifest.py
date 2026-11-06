"""The release manifest ``release.json`` (format ``hcd_priya.release/1``; docs/release_format_h5.md, section "The
release manifest"): one small closed JSON object naming the two parts of an LF+MF release and their published digests,
so that ONE published digest anchors the whole release.

    release_dir/
      release.json     {"format": "hcd_priya.release/1", "release_version": "<text>",
                        "backbone": {"dir": "<name>", "manifest_sha256": "<64 hex>"},
                        "mf": {"dir": "<name>", "record_sha256": "<64 hex>"}}
      <backbone dir>/  the hcd_priya.h5/2 production bundle, whose manifest hashes to manifest_sha256
      <mf dir>/        the hcd_priya.mf_record/1 record, whose mf_record.json hashes to record_sha256

The anchor is the sha256 of ``release.json``. Each ``dir`` is one plain path segment inside the release directory
(never a path, ``..`` or a symlink). The MF part is required: an LF-only bundle is not a release of this format (PI
ruling 2026-11-06a item 2) and keeps loading with ``predictor.load_bundle``. The record pins its backbone's manifest
digest, so a manifest naming a mismatched pair is refused when the parts are loaded (``predictor.load_release``)."""
from __future__ import annotations

import hashlib
import json
import os
import re

from . import safe_io as S

FORMAT = "hcd_priya.release/1"
FILE = "release.json"
KEYS = {"format", "release_version", "backbone", "mf"}
PARTS = {"backbone": ("dir", "manifest_sha256"), "mf": ("dir", "record_sha256")}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PLAIN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")


class ReleaseManifestError(ValueError):
    """A release manifest that may not be written or loaded as asked."""


def check_release_manifest(m):
    """Validate a manifest dict (closed keys, the format, both parts with a plain directory name and a digest)."""
    if not isinstance(m, dict) or set(m) != KEYS:
        got = set(m) if isinstance(m, dict) else set()
        raise ReleaseManifestError(f"release manifest keys must be exactly {sorted(KEYS)} (missing "
                                   f"{sorted(KEYS - got)}, unknown {sorted(got - KEYS)})")
    if m["format"] != FORMAT:
        raise ReleaseManifestError(f"release manifest format {m['format']!r}, expected {FORMAT}")
    if not isinstance(m["release_version"], str) or not m["release_version"].strip():
        raise ReleaseManifestError("release_version must be a non-empty string")
    if m["mf"] is None:
        raise ReleaseManifestError("the release names no MF record: an LF-only bundle is not an LF+MF release (load "
                                   "it with predictor.load_bundle)")
    for part, (dk, hk) in PARTS.items():
        p = m[part]
        if not isinstance(p, dict) or set(p) != {dk, hk}:
            raise ReleaseManifestError(f"{part}: keys must be exactly {sorted((dk, hk))}")
        if not isinstance(p[dk], str) or not _PLAIN.match(p[dk]):
            raise ReleaseManifestError(f"{part}.{dk} must be one plain path segment inside the release, never a path")
        if not isinstance(p[hk], str) or not _HEX64.match(p[hk]):
            raise ReleaseManifestError(f"{part}.{hk} must be 64 lowercase hex digits")
    if m["backbone"]["dir"] == m["mf"]["dir"]:
        raise ReleaseManifestError("the backbone and the MF record are two different directories")
    return m


def _part_dirs(d, m):
    out = {}
    for part in PARTS:
        sub = os.path.join(d, m[part]["dir"])
        if os.path.islink(sub) or not os.path.isdir(sub):
            raise ReleaseManifestError(f"{part}: {m[part]['dir']!r} is not a directory inside the release (a symlink "
                                       "is refused)")
        out[part] = sub
    return out


def read_release_manifest(release_dir, expected_sha256):
    """Read ``release.json`` through a no-follow descriptor, compare its sha256 with the published one BEFORE parsing
    (strict JSON: no duplicate keys, no NaN), validate it, and resolve the two part directories. Returns
    (manifest dict, {"backbone": path, "mf": path})."""
    if not isinstance(expected_sha256, str) or not _HEX64.match(expected_sha256):
        raise ReleaseManifestError("a release loads only with its published sha256 (64 lowercase hex digits)")
    d = os.path.abspath(os.fspath(release_dir))
    if os.path.islink(d) or not os.path.isdir(d):
        raise ReleaseManifestError(f"{release_dir}: not a release directory")
    p = os.path.join(d, FILE)
    try:
        raw = S.read_regular(p, S.MAX_JSON_BYTES)
    except (S.SafeFormatError, OSError) as e:
        raise ReleaseManifestError(f"{release_dir}: no readable {FILE} ({e})") from None
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ReleaseManifestError(f"{p}: sha256 does not match the published release sha256")
    try:
        m = S._json_loads(raw, p)
    except S.SafeFormatError as e:
        raise ReleaseManifestError(str(e)) from None
    check_release_manifest(m)
    return m, _part_dirs(d, m)


def write_release_manifest(release_dir, *, release_version, backbone_dir, backbone_manifest_sha256, mf_dir,
                           mf_record_sha256):
    """Write ``release.json`` ONCE (an existing one is refused) into ``release_dir``, which already holds both parts
    under the given directory names. Returns (path, sha256: the anchor to publish)."""
    m = {"format": FORMAT, "release_version": release_version,
         "backbone": {"dir": backbone_dir, "manifest_sha256": backbone_manifest_sha256},
         "mf": {"dir": mf_dir, "record_sha256": mf_record_sha256}}
    check_release_manifest(m)
    d = os.fspath(release_dir)
    _part_dirs(d, m)
    p = os.path.join(d, FILE)
    if os.path.lexists(p):
        raise FileExistsError(f"{p} exists: a release manifest is written once")
    raw = json.dumps(m, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    return p, S._write_exclusive(p, [raw])
