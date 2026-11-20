"""The public tutorial notebook, notebooks/tutorial/: its builder, its helper cells and its stored files.

The notebook is written by ``notebooks/tutorial/_build_tutorial.py``, which holds every cell as source. The helper
functions the notebook needs are notebook cells; these tests take those cells from the builder
(``helper_source(cell_id)``) and run them in a namespace of their own, so what is tested is the code a reader
runs. Every number in these tests is synthetic.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import http.server
import importlib.util
import json
import pathlib
import re
import sys
import threading

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "notebooks" / "tutorial"
BUILDER = TUTORIAL / "_build_tutorial.py"
NOTEBOOK = TUTORIAL / "hcd_priya_tutorial.ipynb"
MANIFEST = TUTORIAL / "assets" / "ASSETS.json"

TAG_URL = "https://raw.githubusercontent.com/jibanCat/hcd_priya/v1.0.0/"
RELEASE_URL = "https://github.com/jibanCat/hcd_priya/releases/download/v1.0.0/"
COMMIT_URL = re.compile(r"https://raw\.githubusercontent\.com/jibanCat/hcd_priya/[0-9a-f]{40}/")
# strings of a private machine that no public text holds; joined here so that this file does not hold them either
PRIVATE_PATHS = tuple("/" + name + "/" for name in ("home", "nfs", "tmp")) + ("tur" + "bo",)


def _builder():
    spec = importlib.util.spec_from_file_location("_build_tutorial", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _namespace(*cell_ids):
    """The helper cells named, run in order in one namespace, as the notebook runs them."""
    builder = _builder()
    ns = {"__name__": "tutorial_cells"}
    for cell_id in cell_ids:
        exec(compile(builder.helper_source(cell_id), f"<cell {cell_id}>", "exec"), ns)
    return ns


def _notebook_cells(path):
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [(c["cell_type"], c["id"], "".join(c["source"])) for c in doc["cells"]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------ the builder
def test_builder_output_equals_the_committed_cells(tmp_path):
    builder = _builder()
    out = tmp_path / "built.ipynb"
    builder.build(out)
    built = _notebook_cells(out)
    assert built == [(kind, cell_id, source) for kind, cell_id, source in builder.CELLS]
    assert built == _notebook_cells(
        NOTEBOOK
    ), "run notebooks/tutorial/_build_tutorial.py and execute the notebook"
    ids = [cell_id for _, cell_id, _ in built]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[a-z0-9-]{1,64}", cell_id) for cell_id in ids)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (doc["nbformat"], doc["nbformat_minor"]) == (4, 5)
    assert all(
        c["outputs"] == [] and c["execution_count"] is None
        for c in doc["cells"]
        if c["cell_type"] == "code"
    )


def test_helper_source_names_only_code_cells():
    builder = _builder()
    assert (
        builder.helper_source("helpers-assets")
        == dict((i, s) for _, i, s in builder.CELLS)["helpers-assets"]
    )
    with pytest.raises(KeyError):
        builder.helper_source("no-such-cell")
    with pytest.raises(KeyError):
        builder.helper_source("title")  # a text cell


def test_every_code_cell_parses():
    for kind, cell_id, source in _builder().CELLS:
        if kind == "code":
            try:
                ast.parse(source)
            except SyntaxError as exc:  # pragma: no cover - the message is the point
                raise AssertionError(f"cell {cell_id} does not parse: {exc}") from None


def test_no_private_attribute_no_private_path_no_em_dash():
    texts = {f"cell {cell_id}": source for _, cell_id, source in _builder().CELLS}
    for path in (BUILDER, MANIFEST, TUTORIAL / "README.md"):
        texts[path.name] = path.read_text(encoding="utf-8")
    for name, text in texts.items():
        assert "—" not in text, f"{name} holds an em-dash"
        for private in PRIVATE_PATHS:
            assert private not in text, f"{name} holds {private!r}"
    for kind, cell_id, source in _builder().CELLS:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split(".")[-1] for a in node.names]
            for n in names:
                private = n.startswith("_") and not (n.startswith("__") and n.endswith("__"))
                assert not private, f"cell {cell_id} uses the private name {n}"


# ------------------------------------------------------------------------------------------ the stored files
def test_every_asset_url_is_immutable():
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    assert records
    paths = [r["path"] for r in records]
    assert len(set(paths)) == len(paths)
    for r in records:
        assert set(r) == {"path", "sha256", "bytes", "url", "what", "source"}, r["path"]
        assert (
            re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
            and isinstance(r["bytes"], int)
            and r["bytes"] > 0
        )
        url = r["url"]
        assert "/main/" not in url and "/master/" not in url and "/HEAD/" not in url, url
        immutable = url.startswith(TAG_URL) or url.startswith(RELEASE_URL) or COMMIT_URL.match(url)
        assert (
            immutable
        ), f"{r['path']}: {url} is not the tag v1.0.0, the release download or a commit"
        if not url.startswith(RELEASE_URL):
            assert url.endswith("/" + r["path"]), f"{r['path']}: the URL names another path"
            local = (
                ROOT / r["path"]
            )  # a file of this repository: the manifest pins the bytes that are here
            assert local.is_file(), f"{r['path']} is not in the repository"
            data = local.read_bytes()
            assert (_sha256(data), len(data)) == (r["sha256"], r["bytes"]), r["path"]


def test_the_assets_cell_holds_the_manifest():
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    ns = _namespace("helpers-assets")
    assert ns["ASSETS"] == {r["path"]: {"sha256": r["sha256"], "url": r["url"]} for r in records}


# ------------------------------------------------------------------------------------------ tutorial_file
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path):
    """A local web server on a folder of synthetic files: (folder, base URL)."""
    folder = tmp_path / "served"
    folder.mkdir()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(folder))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield folder, f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def cells(tmp_path, monkeypatch):
    """The asset cell with an empty table, a clone folder and a cache folder of its own."""
    ns = _namespace("helpers-assets")
    clone, cache = tmp_path / "clone", tmp_path / "cache"
    clone.mkdir()
    monkeypatch.setenv("HCD_PRIYA_TUTORIAL_CACHE", str(cache))
    ns["ASSETS"] = {}
    ns["LOCAL_ROOTS"] = (clone,)
    return ns, clone, cache


def _files_under(folder):
    return (
        sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
        if folder.exists()
        else []
    )


def test_tutorial_file_uses_a_local_copy_only_with_the_pinned_digest(cells, served):
    ns, clone, cache = cells
    folder, base = served
    good = b"synthetic table, the pinned bytes\n" * 7
    (folder / "data").mkdir()
    (folder / "data" / "table.txt").write_bytes(good)
    ns["ASSETS"]["data/table.txt"] = {"sha256": _sha256(good), "url": base + "data/table.txt"}
    (clone / "data").mkdir()
    # a local copy with the pinned digest is used, and nothing is downloaded
    (clone / "data" / "table.txt").write_bytes(good)
    assert ns["tutorial_file"]("data/table.txt") == clone / "data" / "table.txt"
    assert _files_under(cache) == []
    # a changed local copy is not used: the pinned bytes are downloaded, and the same bytes are read
    (clone / "data" / "table.txt").write_bytes(b"a changed local copy\n")
    got = ns["tutorial_file"]("data/table.txt")
    assert got == cache / "data" / "table.txt" and got.read_bytes() == good
    assert _files_under(cache) == ["data/table.txt"]
    # the cached copy is used the next time, also with the server's file gone
    (folder / "data" / "table.txt").unlink()
    assert ns["tutorial_file"]("data/table.txt") == got
    # a file that the table does not name is refused by name
    with pytest.raises(RuntimeError, match="data/other.txt"):
        ns["tutorial_file"]("data/other.txt")


def test_tutorial_file_refuses_a_wrong_digest_and_leaves_no_partial_file(cells, served):
    ns, clone, cache = cells
    folder, base = served
    good = b"synthetic table, the pinned bytes\n" * 7
    truncated = good[: len(good) // 2]
    (folder / "table.txt").write_bytes(truncated)
    ns["ASSETS"]["table.txt"] = {"sha256": _sha256(good), "url": base + "table.txt"}
    with pytest.raises(RuntimeError) as err:
        ns["tutorial_file"]("table.txt")
    message = str(err.value)
    assert _sha256(good) in message and _sha256(truncated) in message and "table.txt" in message
    assert _files_under(cache) == [], "a refused download left a file in the cache"
    # a cached file that was changed afterwards is not used either: it is replaced by the pinned bytes
    (folder / "table.txt").write_bytes(good)
    assert ns["tutorial_file"]("table.txt").read_bytes() == good
    (cache / "table.txt").write_bytes(truncated)
    assert ns["tutorial_file"]("table.txt").read_bytes() == good
    assert _files_under(cache) == ["table.txt"]


def test_tutorial_file_names_the_file_the_url_and_the_cache_when_the_download_fails(cells, served):
    ns, clone, cache = cells
    folder, base = served
    ns["ASSETS"]["missing.txt"] = {"sha256": _sha256(b"never served"), "url": base + "missing.txt"}
    with pytest.raises(RuntimeError) as err:
        ns["tutorial_file"]("missing.txt")
    message = str(err.value)
    assert "missing.txt" in message and base + "missing.txt" in message and str(cache) in message
    assert _files_under(cache) == []
    # no server at all
    ns["ASSETS"]["unreachable.txt"] = {
        "sha256": _sha256(b"x"),
        "url": "http://127.0.0.1:9/unreachable.txt",
    }
    with pytest.raises(RuntimeError) as err:
        ns["tutorial_file"]("unreachable.txt")
    assert "http://127.0.0.1:9/unreachable.txt" in str(err.value) and str(cache) in str(err.value)
    assert _files_under(cache) == []


# ------------------------------------------------------------------------------------------ the incidence helper
def _figure_script(name):
    scripts = ROOT / "scripts" / "figures"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def test_cumulative_incidence_equals_the_figure_script():
    np = pytest.importorskip("numpy")
    ns = _namespace("helpers-incidence")
    script = _figure_script("fig_dndx_vs_ho21")
    log_nhi = 17.1 + 0.2 * np.arange(30)  # thirty bin centres, as a prediction has
    rng = np.random.default_rng(7)
    cddf = 10.0 ** (-17.0 - 1.5 * (log_nhi - 17.0)) * rng.uniform(
        0.5, 2.0, size=(4, 30)
    )  # synthetic, four rows
    cddf[1, 25:] = np.nan  # bins without a value are left out
    cddf[2, :] = np.nan  # a row with no value at all
    for threshold in (17.5, 17.2, 19.0, 20.37):
        ours = ns["cumulative_incidence"](cddf, log_nhi, threshold)
        assert ours.shape == (4,)
        assert np.array_equal(
            ours, script.cumulative_dndx(cddf, log_nhi, threshold), equal_nan=True
        )
    assert np.isnan(ns["cumulative_incidence"](cddf, log_nhi)[2])
    assert np.array_equal(
        ns["cumulative_incidence"](cddf, log_nhi),
        script.cumulative_dndx(cddf, log_nhi),
        equal_nan=True,
    )
    # the threshold inside a bin counts that bin from the threshold to its upper edge
    one = np.zeros(30)
    one[2] = 1.0  # the bin centred on 17.5, from 17.4 to 17.6
    assert ns["cumulative_incidence"](one, log_nhi, 17.5) == pytest.approx(
        10.0**17.6 - 10.0**17.5, rel=1e-12
    )
    with pytest.raises(ValueError):
        ns["cumulative_incidence"](cddf, log_nhi**1.01)


def test_absorption_distance_equals_the_figure_scripts():
    np = pytest.importorskip("numpy")
    ns = _namespace("helpers-incidence")
    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    try:
        common = _figure_script("figures_common")
    finally:
        sys.path.remove(str(ROOT / "scripts" / "figures"))
    for z1, z2, omega_m in ((2.0, 2.5, 0.3), (3.0, 4.0, 0.279), (1.5, 5.4, 0.25)):
        assert ns["absorption_distance"](z1, z2, omega_m) == pytest.approx(
            common.absorption_distance(z1, z2, omega_m), rel=1e-6
        )
    assert np.isclose(ns["absorption_distance"](2.0, 2.0, 0.3), 0.0)


# ------------------------------------------------------------------------------------------ the committed notebook
def test_the_committed_notebook_is_executed_in_order_without_an_error_or_a_private_path():
    raw = NOTEBOOK.read_text(encoding="utf-8")
    for private in PRIVATE_PATHS:
        assert private not in raw, f"the committed notebook holds {private!r}"
    assert "—" not in raw
    code = [c for c in json.loads(raw)["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(
        range(1, len(code) + 1)
    ), "not executed top to bottom"
    for c in code:
        for output in c["outputs"]:
            assert output["output_type"] != "error", f"cell {c['id']} ended in an error"
            assert (
                output.get("name") != "stderr"
            ), f"cell {c['id']} wrote to stderr: {''.join(output['text'])[:200]}"
