"""The model registry (hcd_analysis.registry): manifest checks, verified download into the cache, offline use and
loading through a format's loader. A fake opener serves synthetic files: no network, no real model, no storage
format."""

import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.request

import pytest

from hcd_analysis import registry as R

BASE = "https://models.example.org/toy-model/1.0.0/"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test caches under tmp_path; nothing reaches the real ~/.cache."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HCD_PRIYA_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("HCD_PRIYA_OFFLINE", raising=False)
    return tmp_path / "cache"


class Server:
    """A fake https server: url -> bytes, counting the bytes each download reads."""

    def __init__(self, files):
        self.files, self.read = dict(files), {}

    def open(self, url):
        server = self

        class Body(io.BytesIO):
            def read(self, n=-1):
                block = super().read(n)
                server.read[url] = server.read.get(url, 0) + len(block)
                return block

        if url not in self.files:
            raise OSError(f"404 {url}")
        return Body(self.files[url])


@pytest.fixture
def server(monkeypatch):
    files = {"manifest.json": b'{"members": 2}\n', "member0.h5": bytes(range(256)) * 8}
    srv = Server({BASE + n: d for n, d in files.items()})
    monkeypatch.setattr(R, "_open_url", srv.open)
    entry = {
        "name": "toy-model",
        "version": "1.0.0",
        "format": "test/1",
        "files": [
            {"path": n, "sha256": _sha(d), "size": len(d), "url": BASE + n}
            for n, d in files.items()
        ],
    }
    return srv, files, entry


def _manifest(*entries):
    return {"schema": R.SCHEMA, "models": list(entries)}


@pytest.fixture
def no_download(monkeypatch):
    def _refuse(url):
        raise AssertionError(f"a download was attempted: {url}")

    monkeypatch.setattr(R, "_open_url", _refuse)


def _root(cache):
    return cache / "toy-model" / "1.0.0"


# ------------------------------------------------------------------------------------------- manifest checks
def test_a_valid_manifest_file_reads_back(tmp_path, server):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(server[2])))
    assert R.read_manifest(path)["models"][0]["name"] == "toy-model"


def _changed(entry, **change):
    e = json.loads(json.dumps(entry))
    f = e["files"][0]
    for key, value in change.items():
        obj, key = (f, key[5:]) if key.startswith("file_") else (e, key)
        if value is None:
            obj.pop(key)
        else:
            obj[key] = value
    return e


@pytest.mark.parametrize(
    "change",
    [
        dict(name=None),
        dict(format=None),
        dict(files=[]),
        dict(extra="typo"),
        dict(format=""),
        dict(name="../x"),
        dict(name=".hidden"),
        dict(version="1/0"),
        dict(version=""),
        dict(name="a" * 101),
        dict(file_sha256="0" * 63),
        dict(file_sha256="A" * 64),
        dict(file_size=-1),
        dict(file_size=True),
        dict(file_size=1.0),
        dict(file_size=R.MAX_FILE_BYTES + 1),
        dict(file_url=None),
        dict(file_extra=1),
        dict(file_path="/etc/passwd"),
        dict(file_path="../escape"),
        dict(file_path="sub/a"),
        dict(file_path="a\\b"),
        dict(file_path="."),
        dict(file_path=""),
        dict(file_path=".partial"),
        dict(file_url="http://example.org/a"),
        dict(file_url="file:///etc/passwd"),
        dict(file_url="ftp://x.org/a"),
        dict(file_url="data:,x"),
        dict(file_url="https://"),
        dict(file_url="https://x.org/" + "a" * 2048),
    ],
)
def test_a_bad_manifest_is_refused(server, change):
    with pytest.raises(R.ManifestError):
        R.validate_manifest(_manifest(_changed(server[2], **change)))


def test_duplicates_unknown_schema_and_size_caps_are_refused(server):
    entry = server[2]
    dup = json.loads(json.dumps(entry))
    dup["files"].append(dict(dup["files"][0], path="MANIFEST.json"))  # the same name, ignoring case
    big = json.loads(json.dumps(entry))
    big["files"] = [dict(entry["files"][0], path=f"f{i}", size=R.MAX_FILE_BYTES) for i in range(5)]
    for doc in (
        _manifest(dup),
        _manifest(entry, entry),
        _manifest(big),
        {"schema": "other/1", "models": [entry]},
        {"schema": R.SCHEMA, "models": [entry], "extra": 1},
    ):
        with pytest.raises(R.ManifestError):
            R.validate_manifest(doc)


@pytest.mark.parametrize(
    "text, message",
    [
        (
            '{"schema": "x", "schema": "hcd_priya.registry/1", "models": []}',
            "duplicate key",
        ),  # the last one is valid
        ('{"schema": "hcd_priya.registry/1", "models": [], "models": [NaN]}', "duplicate key|NaN"),
        ('{"schema": "hcd_priya.registry/1", "models": [Infinity]}', "Infinity is not allowed"),
    ],
)
def test_a_manifest_file_with_duplicate_keys_or_non_finite_numbers_is_refused(
    tmp_path, text, message
):
    path = tmp_path / "manifest.json"
    path.write_text(text)
    with pytest.raises(R.ManifestError, match=message):
        R.read_manifest(path)


def test_an_unknown_model_error_lists_the_models_that_exist(server):
    with pytest.raises(R.ManifestError, match="toy-model 1.0.0"):
        R.load(_manifest(server[2]), "toy-model", "2.0.0")


# ------------------------------------------------------------------------------------------- fetch and cache
def test_fetch_downloads_verifies_and_caches_only_the_listed_files(isolated_cache, server):
    _, files, entry = server
    root = R.fetch(entry)
    assert root == _root(isolated_cache)
    assert sorted(os.listdir(root)) == sorted(files)  # nothing else, not even a temporary file
    for name, data in files.items():
        assert (root / name).read_bytes() == data


def test_a_second_fetch_reuses_the_verified_cache(isolated_cache, server, no_download):
    _, files, entry = server
    root = _root(isolated_cache)
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    assert R.fetch(entry) == root


def test_a_same_size_corruption_is_caught_by_the_hash_and_downloaded_again(isolated_cache, server):
    _, files, entry = server
    root = R.fetch(entry)
    data = bytearray(files["member0.h5"])
    data[100] ^= 1  # one bit, same size: only the sha256 can tell
    (root / "member0.h5").write_bytes(bytes(data))
    with pytest.warns(UserWarning, match="differs from the manifest"):
        R.fetch(entry)
    assert (root / "member0.h5").read_bytes() == files["member0.h5"]


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_a_symlink_or_fifo_in_place_of_a_cached_file_is_never_trusted(
    isolated_cache, server, tmp_path, kind
):
    _, files, entry = server
    root = R.fetch(entry)
    (root / "manifest.json").unlink()
    if kind == "symlink":
        other = tmp_path / "elsewhere.json"
        other.write_bytes(files["manifest.json"])  # the right bytes, outside the cache
        (root / "manifest.json").symlink_to(other)
    else:
        os.mkfifo(root / "manifest.json")  # hashing it must not block
    R.fetch(entry)
    assert (
        not (root / "manifest.json").is_symlink()
        and (root / "manifest.json").read_bytes() == files["manifest.json"]
    )


def test_a_symlinked_model_directory_is_refused(isolated_cache, server, tmp_path):
    _, _, entry = server
    (isolated_cache / "toy-model").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (isolated_cache / "toy-model" / "1.0.0").symlink_to(
        tmp_path / "outside", target_is_directory=True
    )
    with pytest.raises(R.RegistryError, match="not a plain directory"):
        R.fetch(entry)
    assert not list((tmp_path / "outside").iterdir())


@pytest.mark.parametrize(
    "what, message",
    [("content", "sha256"), ("longer", "more than"), ("shorter", "bytes, the manifest lists")],
)
def test_a_source_that_does_not_match_the_manifest_leaves_nothing(
    isolated_cache, server, what, message
):
    srv, files, entry = server
    data = files["member0.h5"]
    srv.files[BASE + "member0.h5"] = {
        "content": data[::-1],
        "longer": data + b"x",
        "shorter": data[:-1],
    }[what]
    with pytest.raises(R.IntegrityError, match="member0.h5.*" + message):
        R.fetch(entry)
    root = _root(isolated_cache)
    assert not os.path.lexists(root / "member0.h5")
    assert not list((isolated_cache / ".partial" / "toy-model" / "1.0.0").iterdir())


def test_an_oversized_source_is_read_only_one_byte_past_the_listed_size(isolated_cache, server):
    srv, files, entry = server
    srv.files[BASE + "member0.h5"] = files["member0.h5"] + b"\0" * (10 << 20)
    with pytest.raises(R.IntegrityError, match="more than"):
        R.fetch(entry)
    assert srv.read[BASE + "member0.h5"] == len(files["member0.h5"]) + 1


def test_only_verified_files_are_moved_into_the_model_directory(
    isolated_cache, server, monkeypatch
):
    srv, files, entry = server
    moved, real_replace = [], os.replace

    def spy(src, dst):
        moved.append((open(src, "rb").read(), os.path.basename(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(R.os, "replace", spy)
    R.fetch(entry)
    srv.files[BASE + "member0.h5"] = b"wrong"
    (_root(isolated_cache) / "member0.h5").write_bytes(b"stale")
    with pytest.raises(R.IntegrityError):
        R.fetch(entry)
    assert sorted(moved, key=lambda m: m[1]) == sorted(
        ((d, n) for n, d in files.items()), key=lambda m: m[1]
    )
    assert (_root(isolated_cache) / "manifest.json").read_bytes() == files["manifest.json"]
    assert not os.path.lexists(
        _root(isolated_cache) / "member0.h5"
    )  # the stale copy is removed, not kept


def test_a_stale_partial_download_does_not_touch_the_model_directory(isolated_cache, server):
    _, files, entry = server
    partial = isolated_cache / ".partial" / "toy-model" / "1.0.0"
    partial.mkdir(parents=True)
    (partial / "member0.h5.abandoned").write_bytes(b"half")  # left by a killed job
    root = R.fetch(entry)
    assert sorted(os.listdir(root)) == sorted(files)


def test_offline_never_downloads_and_names_what_is_missing_or_bad(
    isolated_cache, server, no_download, monkeypatch
):
    _, files, entry = server
    with pytest.raises(R.OfflineError, match="manifest.json.*missing"):
        R.fetch(entry, offline=True)
    root = _root(isolated_cache)
    for name, data in files.items():
        (root / name).write_bytes(data)
    (root / "member0.h5").write_bytes(b"corrupt")
    monkeypatch.setenv("HCD_PRIYA_OFFLINE", "1")
    with pytest.raises(R.OfflineError, match="member0.h5.*not verified"):
        R.fetch(entry)


def test_fetch_validates_the_entry_it_is_given(server):
    with pytest.raises(R.ManifestError):
        R.fetch(_changed(server[2], file_path="../escape"))


def test_redirects_to_anything_but_https_are_refused():
    req = urllib.request.Request("https://models.example.org/a")
    for url in ("http://models.example.org/a", "file:///etc/passwd", "ftp://models.example.org/a"):
        with pytest.raises(R.RegistryError, match="only https"):
            R._HttpsOnlyRedirect().redirect_request(req, None, 302, "Found", {}, url)


def test_downloads_go_through_an_opener_that_follows_only_https_redirects(monkeypatch):
    """The opener _open_url uses carries exactly one redirect handler, the https-only one (urllib's default handler
    is replaced, not kept beside it); _open_url opens through that opener; and a redirect to https is followed.
    """
    redirects = [h for h in R._OPENER.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
    assert len(redirects) == 1 and type(redirects[0]) is R._HttpsOnlyRedirect
    seen = []

    class _Opener:
        def open(self, url, timeout):
            seen.append((url, timeout))
            return "response"

    monkeypatch.setattr(R, "_OPENER", _Opener())
    assert R._open_url("https://models.example.org/a") == "response"
    assert seen == [("https://models.example.org/a", R.TIMEOUT_S)]
    req = urllib.request.Request("https://models.example.org/a")
    new = R._HttpsOnlyRedirect().redirect_request(
        req, None, 302, "Found", {}, "https://cdn.example.org/a"
    )
    assert isinstance(new, urllib.request.Request) and new.full_url == "https://cdn.example.org/a"


def test_the_cache_location_follows_the_environment(tmp_path, monkeypatch):
    assert R.cache_dir() == tmp_path / "cache"
    monkeypatch.delenv("HCD_PRIYA_CACHE")
    assert R.cache_dir() == tmp_path / "home" / ".cache" / "hcd_priya"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert R.cache_dir() == tmp_path / "xdg" / "hcd_priya"


# ------------------------------------------------------------------------------------------- loading
def test_load_reads_a_manifest_file_and_hands_the_verified_directory_to_the_format_loader(
    isolated_cache, server, monkeypatch, tmp_path
):
    _, _, entry = server
    seen = {}

    def loader(root, e):
        seen["files"] = sorted(os.listdir(root))
        seen["entry"] = e
        return "model"

    monkeypatch.setattr(R, "_FORMATS", {"test/1": loader})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(entry)))
    assert R.load(path, "toy-model", "1.0.0") == "model"
    assert seen["files"] == ["manifest.json", "member0.h5"] and seen["entry"]["format"] == "test/1"


def test_a_format_loader_named_by_module_is_imported_only_when_used(
    isolated_cache, server, monkeypatch
):
    monkeypatch.setattr(R, "_FORMATS", {"test/1": "json:dumps"})
    assert R._format_loader("test/1") is json.dumps


def test_a_format_without_a_loader_is_refused_before_any_download(server, no_download, monkeypatch):
    monkeypatch.setattr(R, "_FORMATS", {"other/1": lambda root, e: None})
    with pytest.raises(R.RegistryError, match=r"test/1.*other/1"):
        R.load(_manifest(server[2]), "toy-model", "1.0.0")


def test_the_loader_is_not_called_when_fetch_fails(server, monkeypatch):
    srv, _, entry = server
    srv.files[BASE + "member0.h5"] = b"wrong"
    called = []
    monkeypatch.setattr(R, "_FORMATS", {"test/1": lambda root, e: called.append(root)})
    with pytest.raises(R.IntegrityError):
        R.load(_manifest(entry), "toy-model", "1.0.0")
    assert called == []


def test_no_format_is_supported_yet_and_nothing_heavy_or_pickle_is_imported():
    """No release format is approved, so no loader is mapped; importing the registry pulls in no array library and
    no pickle (checked in a fresh interpreter, so other tests' imports do not hide one)."""
    assert R.supported_formats() == []
    code = (
        "import sys, hcd_analysis.registry; "
        "print(sorted(m for m in ('jax', 'numpy', 'h5py', 'safetensors', 'pickle', 'cloudpickle', 'dill') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-E", "-s", "-B", "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(R.__file__)),
    )  # the directory holding hcd_analysis
    assert out.stdout.strip() == "[]"
