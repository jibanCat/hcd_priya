"""The release manifest ``release.json`` (format ``hcd_priya.release/1``;
docs/release_format_h5.md, section "The release manifest"): one small closed JSON object naming the parts of an LF+MF
release and their published digests, so that ONE published digest anchors the whole release.

    release_dir/
      release.json     {"format": "hcd_priya.release/1", "release_version": "<text>",
                        "backbone": {"dir": "<name>", "manifest_sha256": "<64 hex>"},
                        "mf": {"dir": "<name>", "record_sha256": "<64 hex>"}}
      <backbone dir>/  the hcd_priya.h5/2 production bundle, whose manifest hashes to manifest_sha256
      <mf dir>/        the hcd_priya.mf_record/1 record, whose mf_record.json hashes to record_sha256

The two parts above are required. A release may also name optional parts, from the closed list ``OPTIONAL_PARTS``,
each a directory with the digest of its record:

      release.json     {..., "incidence": {"dir": "<name>", "record_sha256": "<64 hex>"}}
      <incidence dir>/ the hcd_priya.incidence_mf_record/1 record (incidence_mf.py): the MF correction of dN/dX

There is one format. A manifest without an optional part is byte for byte what it was before optional parts
existed. A part outside the list is refused, so a reader that does not know a part never ignores it: a reader from
before optional parts existed refuses a manifest that names one.

The anchor is the sha256 of ``release.json``. Each ``dir`` is one plain path segment inside the release directory
(never a path, ``..`` or a symlink). The MF part is required: an LF-only bundle is not a release of this format (PI
ruling 2026-11-06a item 2) and keeps loading with ``predictor.load_bundle``. The record pins its backbone's manifest
digest, so a manifest naming a mismatched pair is refused when the parts are loaded (``predictor.load_release``).
"""

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
# The optional parts a release may name (a closed list; a later part is added here).
OPTIONAL_PARTS = {"incidence": ("dir", "record_sha256")}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PLAIN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")


class ReleaseManifestError(ValueError):
    """A release manifest that may not be written or loaded as asked."""


def check_release_manifest(m):
    """Validate a manifest dict: closed keys, the format, every part with a plain directory name and a digest.
    The keys are the four required ones and any of the optional parts of ``OPTIONAL_PARTS``; any other key is
    refused.
    """
    got = set(m) if isinstance(m, dict) else set()
    fmt = m.get("format") if isinstance(m, dict) else None
    optional = set(OPTIONAL_PARTS)
    if not KEYS <= got or got - KEYS - optional:
        raise ReleaseManifestError(
            f"release manifest keys must be {sorted(KEYS)}, with optional parts among {sorted(optional)} "
            f"(missing {sorted(KEYS - got)}, unknown {sorted(got - KEYS - optional)})"
        )
    if fmt != FORMAT:
        raise ReleaseManifestError(f"release manifest format {fmt!r}, expected {FORMAT}")
    if not isinstance(m["release_version"], str) or not m["release_version"].strip():
        raise ReleaseManifestError("release_version must be a non-empty string")
    if m["mf"] is None:
        raise ReleaseManifestError(
            "the release names no MF record: an LF-only bundle is not an LF+MF release (load "
            "it with predictor.load_bundle)"
        )
    for part, (dk, hk) in _parts(m).items():
        p = m[part]
        if not isinstance(p, dict) or set(p) != {dk, hk}:
            raise ReleaseManifestError(f"{part}: keys must be exactly {sorted((dk, hk))}")
        if not isinstance(p[dk], str) or not _PLAIN.match(p[dk]):
            raise ReleaseManifestError(
                f"{part}.{dk} must be one plain path segment inside the release, never a path"
            )
        if not isinstance(p[hk], str) or not _HEX64.match(p[hk]):
            raise ReleaseManifestError(f"{part}.{hk} must be 64 lowercase hex digits")
        if (
            type(p[dk]) is not str or type(p[hk]) is not str
        ):  # a parsed manifest always holds exact str
            raise ReleaseManifestError(
                f"{part}: {dk} and {hk} must be exact str (a str subclass is refused)"
            )
    dirs = [m[part]["dir"] for part in _parts(m)]
    if len(set(dirs)) != len(dirs):
        raise ReleaseManifestError(
            "the backbone and the MF record are two different directories, and so is every other part"
        )
    return m


def _parts(m):
    """The parts a manifest names, with their two keys: the required ones, then the optional ones it holds."""
    return dict(PARTS, **{k: v for k, v in OPTIONAL_PARTS.items() if k in m})


def _part_dirs(d, m):
    out = {}
    for part in _parts(m):
        sub = os.path.join(d, m[part]["dir"])
        if os.path.islink(sub) or not os.path.isdir(sub):
            raise ReleaseManifestError(
                f"{part}: {m[part]['dir']!r} is not a directory inside the release (a symlink "
                "is refused)"
            )
        out[part] = sub
    return out


def read_release_manifest(release_dir, expected_sha256):
    """Read ``release.json`` through a no-follow descriptor, compare its sha256 with the published one BEFORE parsing
    (strict JSON: no duplicate keys, no NaN), validate it, and resolve the part directories. Returns
    (manifest dict, {"backbone": path, "mf": path, and "incidence": path when the manifest names it}).
    """
    if type(expected_sha256) is not str or not _HEX64.match(
        expected_sha256
    ):  # exact str (delta review R1-2)
        raise ReleaseManifestError(
            "a release loads only with its published sha256 (64 lowercase hex digits, an "
            "exact str)"
        )
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


def write_release_manifest(
    release_dir,
    *,
    release_version,
    backbone_dir,
    backbone_manifest_sha256,
    mf_dir,
    mf_record_sha256,
    incidence_dir=None,
    incidence_record_sha256=None,
):
    """Write ``release.json`` ONCE (an existing one is refused) into ``release_dir``, which already holds every part
    under the given directory names. With ``incidence_dir`` and ``incidence_record_sha256`` (both or neither) the
    manifest also names the incidence MF record, an optional part; without them it is byte for byte what it was
    before optional parts existed. Returns (path, sha256: the anchor to publish)."""
    if (incidence_dir is None) != (incidence_record_sha256 is None):
        raise ReleaseManifestError(
            "give both incidence_dir and incidence_record_sha256, or neither"
        )
    m = {
        "format": FORMAT,
        "release_version": release_version,
        "backbone": {"dir": backbone_dir, "manifest_sha256": backbone_manifest_sha256},
        "mf": {"dir": mf_dir, "record_sha256": mf_record_sha256},
    }
    if incidence_dir is not None:
        m["incidence"] = {"dir": incidence_dir, "record_sha256": incidence_record_sha256}
    check_release_manifest(m)
    d = os.fspath(release_dir)
    _part_dirs(d, m)
    p = os.path.join(d, FILE)
    if os.path.lexists(p):
        raise FileExistsError(f"{p} exists: a release manifest is written once")
    raw = json.dumps(m, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    return p, S._write_exclusive(p, [raw])
