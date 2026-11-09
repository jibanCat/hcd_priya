"""Fetch, verify and cache released models by name and version, then load them (provisional interface).

A manifest (JSON, schema ``hcd_priya.registry/1``) lists each model's files with their sha256, size and URL::

    {"schema": "hcd_priya.registry/1",
     "models": [{"name": "...", "version": "...", "format": "...",
                 "files": [{"path": "...", "sha256": "...", "size": 123, "url": "https://..."}]}]}

One call gets a model: ``registry.load("manifest.json", name, version)`` fetches and verifies its files, then hands
the verified directory to the loader of the model's format. ``registry.fetch`` alone gives the verified directory.

The registry knows files, checksums, versions and the cache, and nothing about any file format. Formats are mapped
to loaders by ``_FORMATS``, in code, never from a manifest; the map is empty because no release format is approved
yet. A format's loader gets ``(root, entry)`` and must itself check every byte it parses against the digests in
``entry`` (for a bundle with its own manifest, that manifest's sha256 is the anchor), so files changed between the
registry's check and the loader's read are refused. Nothing here reads a pickle. No manifest ships with the package:
where released models are hosted is not decided yet.

- Cache: ``$HCD_PRIYA_CACHE``, else ``$XDG_CACHE_HOME/hcd_priya``, else ``~/.cache/hcd_priya``. A model's files are
  ``<cache>/<name>/<version>/<file>``; removing that directory removes the model, and the next fetch downloads it
  again. Downloads in progress live under ``<cache>/.partial/``, never in a model's directory.
- Offline: ``HCD_PRIYA_OFFLINE=1`` (or ``offline=True``) never downloads; a missing or bad file is an error naming it.
- Only ``https`` URLs, and https redirects, are followed. A file is moved into the model's directory only after its
  size and sha256 match; cached files are hashed again on every fetch; a symlink or any non-regular file in the cache
  is never trusted.
"""
from __future__ import annotations

import errno
import hashlib
import importlib
import json
import logging
import os
import re
import stat
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

SCHEMA = "hcd_priya.registry/1"
MAX_FILE_BYTES = 1 << 32
MAX_MODEL_BYTES = 1 << 34
MAX_FILES = 1000
TIMEOUT_S = 60
_CHUNK = 1 << 20
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,99}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = {"schema", "models"}
_MODEL_KEYS = {"name", "version", "format", "files"}
_FILE_KEYS = {"path", "sha256", "size", "url"}

# format id -> "module:function" (imported only when a model of that format is loaded) or a callable. Empty until
# the PI approves a release format.
_FORMATS = {}

log = logging.getLogger(__name__)


class RegistryError(Exception):
    """Any registry failure."""


class ManifestError(RegistryError):
    """The manifest is malformed, or does not list the requested model."""


class IntegrityError(RegistryError):
    """Downloaded bytes do not match the manifest's size or sha256."""


class OfflineError(RegistryError):
    """Offline, and a file is missing from the cache or does not verify."""


# ------------------------------------------------------------------------------------------------- manifest
def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ManifestError(f"manifest: duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name):
    raise ManifestError(f"manifest: {name} is not allowed")


def _keys(obj, required, where):
    if not isinstance(obj, dict):
        raise ManifestError(f"{where}: expected an object")
    missing, unknown = required - obj.keys(), obj.keys() - required
    if missing or unknown:
        raise ManifestError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _segment(value, what, where):
    if not isinstance(value, str) or not _SEGMENT.match(value) or ".." in value:
        raise ManifestError(f"{where}: {what} {value!r} is not a plain name (1-100 of A-Za-z0-9._+-, no '..')")


def _url(value, where):
    parts = urllib.parse.urlsplit(value) if isinstance(value, str) and len(value) <= 2048 else None
    if parts is None or parts.scheme != "https" or not parts.netloc:
        raise ManifestError(f"{where}: url {value!r} must be an https URL")


def validate_manifest(doc):
    """Check a manifest completely (keys, names, files, checksums, sizes, URLs); return it unchanged."""
    _keys(doc, _MANIFEST_KEYS, "manifest")
    if doc["schema"] != SCHEMA:
        raise ManifestError(f"manifest: schema {doc['schema']!r}, expected {SCHEMA!r}")
    if not isinstance(doc["models"], list):
        raise ManifestError("manifest: models must be a list")
    seen = set()
    for i, m in enumerate(doc["models"]):
        where = f"models[{i}]"
        _keys(m, _MODEL_KEYS, where)
        _segment(m["name"], "name", where)
        _segment(m["version"], "version", where)
        if not isinstance(m["format"], str) or not 0 < len(m["format"]) <= 100:
            raise ManifestError(f"{where}: format must be a string of 1-100 characters")
        if (m["name"], m["version"]) in seen:
            raise ManifestError(f"{where}: {m['name']} {m['version']} is listed twice")
        seen.add((m["name"], m["version"]))
        if not isinstance(m["files"], list) or not 0 < len(m["files"]) <= MAX_FILES:
            raise ManifestError(f"{where}: files must be a list of 1-{MAX_FILES} entries")
        names, total = set(), 0
        for j, f in enumerate(m["files"]):
            fwhere = f"{where}.files[{j}]"
            _keys(f, _FILE_KEYS, fwhere)
            _segment(f["path"], "path", fwhere)                     # a bundle is flat: one plain file name
            if f["path"].lower() in names:
                raise ManifestError(f"{fwhere}: path {f['path']!r} is listed twice (ignoring case)")
            names.add(f["path"].lower())
            if not isinstance(f["sha256"], str) or not _SHA256.match(f["sha256"]):
                raise ManifestError(f"{fwhere}: sha256 must be 64 lower-case hex digits")
            size = f["size"]
            if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_FILE_BYTES:
                raise ManifestError(f"{fwhere}: size must be an integer in [0, {MAX_FILE_BYTES}]")
            total += size
            _url(f["url"], fwhere)
        if total > MAX_MODEL_BYTES:
            raise ManifestError(f"{where}: {total} bytes in all, more than {MAX_MODEL_BYTES}")
    return doc


def read_manifest(path):
    """Read and validate a manifest from a local JSON file (duplicate keys, NaN and Infinity are refused)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    doc = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_no_constant)
    log.info("manifest %s (sha256 %s)", path, hashlib.sha256(raw).hexdigest())
    return validate_manifest(doc)


def _select(manifest, name, version):
    for m in manifest["models"]:
        if m["name"] == name and m["version"] == version:
            return m
    listed = ", ".join(f"{m['name']} {m['version']}" for m in manifest["models"]) or "none"
    raise ManifestError(f"no model {name!r} version {version!r} in the manifest (listed: {listed})")


# ------------------------------------------------------------------------------------------------- cache
def cache_dir():
    """The cache root: $HCD_PRIYA_CACHE, else $XDG_CACHE_HOME/hcd_priya, else ~/.cache/hcd_priya."""
    if os.environ.get("HCD_PRIYA_CACHE"):
        return Path(os.environ["HCD_PRIYA_CACHE"])
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "hcd_priya"


def _real_dir(path):
    """Create path if needed; refuse it if it is a symlink or not a directory."""
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise RegistryError(f"{path} is not a plain directory (a symlink or a file): refusing to use it")
    return path


def _cached_digest(target, size):
    """The sha256 of target if it is a regular file of the right size (never following a symlink), else None."""
    try:
        if not stat.S_ISREG(os.lstat(target).st_mode):
            return None
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ELOOP):
            return None
        raise
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size != size:
            return None
        h = hashlib.sha256()
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise RegistryError(f"refused a redirect from {req.full_url} to {newurl}: only https is followed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsOnlyRedirect)


def _open_url(url):
    return _OPENER.open(url, timeout=TIMEOUT_S)


def _download(f, partial_dir, target):
    """Stream f["url"] into a temporary file under partial_dir; move it to target only if size and sha256 match."""
    fd, tmp = tempfile.mkstemp(prefix=f["path"] + ".", dir=partial_dir)
    try:
        h, n = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as out, _open_url(f["url"]) as src:
            while True:
                block = src.read(min(_CHUNK, f["size"] + 1 - n))
                if not block:
                    break
                n += len(block)
                if n > f["size"]:
                    raise IntegrityError(f"{f['path']} from {f['url']}: more than the {f['size']} bytes listed")
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if n != f["size"]:
            raise IntegrityError(f"{f['path']} from {f['url']}: {n} bytes, the manifest lists {f['size']}")
        if h.hexdigest() != f["sha256"]:
            raise IntegrityError(f"{f['path']} from {f['url']}: sha256 {h.hexdigest()}, the manifest lists "
                                 f"{f['sha256']} ({n} bytes)")
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def fetch(entry, cache=None, offline=None):
    """Make every file of a manifest entry present and verified in the cache; return the model's directory.

    The entry is validated again. Cached files are hashed again. Online, a file that is missing or does not verify
    is removed and downloaded again (with a warning when a cached copy changed), so a failed download leaves nothing
    at its path; offline, it is reported. A file reaches the model's directory only after it verifies."""
    validate_manifest({"schema": SCHEMA, "models": [entry]})
    if offline is None:
        offline = os.environ.get("HCD_PRIYA_OFFLINE") == "1"
    base = Path(cache) if cache is not None else cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    root = _real_dir(_real_dir(base / entry["name"]) / entry["version"])
    partial = None
    for f in entry["files"]:
        target = root / f["path"]
        digest = _cached_digest(target, f["size"])
        if digest == f["sha256"]:
            continue
        if offline:
            state = "missing from" if not os.path.lexists(target) else "not verified in"
            raise OfflineError(f"{f['path']} of {entry['name']} {entry['version']} is {state} the cache {root} "
                               "and downloads are off (HCD_PRIYA_OFFLINE=1 or offline=True)")
        if digest is not None:
            warnings.warn(f"{target}: the cached copy (sha256 {digest}) differs from the manifest (sha256 "
                          f"{f['sha256']}); downloading it again", stacklevel=2)
        if os.path.lexists(target):
            if stat.S_ISDIR(os.lstat(target).st_mode):
                raise RegistryError(f"{target} is a directory: refusing to replace it")
            os.unlink(target)
        if partial is None:
            partial = _real_dir(_real_dir(_real_dir(base / ".partial") / entry["name"]) / entry["version"])
        _download(f, partial, target)
    return root


# ------------------------------------------------------------------------------------------------- loading
def supported_formats():
    return sorted(_FORMATS)


def _format_loader(fmt):
    spec = _FORMATS.get(fmt)
    if spec is None:
        raise RegistryError(f"no loader for format {fmt!r} (supported: {', '.join(supported_formats()) or 'none'})")
    if callable(spec):
        return spec
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func)


def load(manifest, name, version, cache=None, offline=None):
    """Fetch and verify a model, then load it with its format's loader. manifest: a path or a validated dict."""
    if not isinstance(manifest, dict):
        manifest = read_manifest(manifest)
    entry = _select(manifest, name, version)
    loader = _format_loader(entry["format"])
    root = fetch(entry, cache, offline)
    log.info("loading %s %s (format %s) from %s", name, version, entry["format"], root)
    return loader(root, entry)
