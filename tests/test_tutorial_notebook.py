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


# ------------------------------------------------------------------------------------------ the older notebooks
LEGACY = ROOT / "notebooks" / "tutorials"
LEGACY_BANNER = (
    "> Legacy internal research workflows. These notebooks need PRIYA simulation outputs that are not distributed. "
    "They are not supported public tutorials. The public tutorial is `notebooks/tutorial/`."
)


def test_the_five_legacy_notebooks_and_their_readme_carry_the_banner():
    notebooks = sorted(LEGACY.glob("*.ipynb"))
    assert [p.name[:2] for p in notebooks] == ["00", "01", "02", "03", "04"]
    for path in notebooks:
        first = json.loads(path.read_text(encoding="utf-8"))["cells"][0]
        assert first["cell_type"] == "markdown" and first["id"] == "legacy-banner", path.name
        assert "".join(first["source"]) == LEGACY_BANNER, path.name
    readme = (LEGACY / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(
        LEGACY_BANNER + "\n\n# "
    ), "the README of the older notebooks does not begin with the banner"
    # the script that writes the five notebooks puts the same cell first, so a rebuild keeps it
    builder_text = (LEGACY / "_build_notebooks.py").read_text(encoding="utf-8")
    assert 'new_markdown_cell(LEGACY_BANNER, id="legacy-banner")] + cells' in builder_text
    scope = {}
    start = builder_text.index("LEGACY_BANNER = (")
    exec(builder_text[start : builder_text.index(")\n", start) + 2], scope)
    assert scope["LEGACY_BANNER"] == LEGACY_BANNER
    assert (TUTORIAL / "hcd_priya_tutorial.ipynb").is_file()  # what the banner points to exists


# ------------------------------------------------------------------------------------------ released code, run as it is
FORBIDDEN_IN_A_NAME = ("sampl", "chain", "metropolis", "posterior", "mcmc")
FORBIDDEN_IMPORTS = (
    "cobaya",
    "emcee",
    "getdist",
    "numpyro",
    "blackjax",
    "dynesty",
    "pymc",
    "scipy.optimize",
)


def test_the_notebook_defines_no_inference_code_of_its_own():
    """The notebook is a tutorial of the published package: it calls the package, the released figure scripts and
    the released example. It defines no sampler, no posterior and no chain code, and imports no sampler or
    optimiser; the removed first draft must not come back."""
    for kind, cell_id, source in _builder().CELLS:
        if kind != "code":
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bad = [word for word in FORBIDDEN_IN_A_NAME if word in node.name.lower()]
                assert not bad, f"cell {cell_id} defines {node.name}"
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            for module in modules:
                assert not any(
                    module == bad or module.startswith(bad + ".") for bad in FORBIDDEN_IMPORTS
                ), f"cell {cell_id} imports {module}"
    assert sorted(p.name for p in TUTORIAL.glob("*.py")) == [
        "_build_tutorial.py"
    ], "a second script beside the builder"


@pytest.fixture
def released(cells, monkeypatch, tmp_path):
    """The asset cell and the cell that runs released code, in a working folder of their own."""
    ns, clone, cache = cells
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)  # the cell makes its output folder in the working folder
    exec(
        compile(_builder().helper_source("helpers-released"), "<cell helpers-released>", "exec"), ns
    )
    assert (work / "tutorial_output").is_dir()
    return ns, clone, cache, work


def _pin(ns, clone, relpath, data):
    """A synthetic file of the clone, pinned in the table as the tag's files are."""
    path = clone / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    ns["ASSETS"][relpath] = {"sha256": _sha256(data), "url": "http://127.0.0.1:9/" + relpath}


def test_a_released_script_runs_unchanged_from_one_folder_with_its_pinned_files(released):
    ns, clone, cache, work = released
    script = (
        b"import pathlib, sys\n"
        b"root = pathlib.Path(__file__).resolve().parents[2]\n"  # as the figure scripts find their data
        b"print('table:', (root / 'docs' / 'data' / 'table.txt').read_text().strip())\n"
        b"print('   a detail line')\n"
        b"print('arguments:', sys.argv[1:])\n"
        b"pathlib.Path(sys.argv[2]).write_text('drawn')\n"
    )
    _pin(ns, clone, "scripts/figures/draw.py", script)
    _pin(ns, clone, "docs/data/table.txt", b"seven synthetic numbers\n")
    printed = ns["run_released_script"](
        "scripts/figures/draw.py",
        ["--out", "tutorial_output/synthetic.png"],
        needs=("docs/data/table.txt",),
    )
    assert printed.splitlines() == [
        "table: seven synthetic numbers",
        "   a detail line",
        "arguments: ['--out', 'tutorial_output/synthetic.png']",
    ]
    assert (
        work / "tutorial_output" / "synthetic.png"
    ).read_text() == "drawn"  # relative to the working folder
    # the script ran from the cache folder, where it and its data stand at their repository paths, as pinned
    assert (cache / "scripts" / "figures" / "draw.py").read_bytes() == script
    assert (cache / "docs" / "data" / "table.txt").read_bytes() == b"seven synthetic numbers\n"
    # a changed copy in the clone is not what runs: the pinned bytes in the cache are
    (clone / "docs" / "data" / "table.txt").write_bytes(b"changed in the clone\n")
    (cache / "docs" / "data" / "table.txt").write_bytes(b"changed in the cache\n")
    with pytest.raises(
        RuntimeError, match="docs/data/table.txt"
    ):  # neither copy is the pinned one, and no server
        ns["run_released_script"](
            "scripts/figures/draw.py", ["--out", "x"], needs=("docs/data/table.txt",)
        )
    # a script that fails stops the notebook with what it wrote
    _pin(
        ns,
        clone,
        "scripts/figures/fails.py",
        b"import sys\nsys.stderr.write('no such table\\n')\nsys.exit(3)\n",
    )
    with pytest.raises(RuntimeError, match="ended with code 3") as err:
        ns["run_released_script"]("scripts/figures/fails.py", [], needs=())
    assert "no such table" in str(err.value)


def test_the_output_of_a_script_is_shown_without_its_detail_lines(released, capsys):
    ns = released[0]
    capsys.readouterr()
    ns["show_script_output"](
        "first line\n   design point 3: a flag\n   design point 4: a flag\n\nwrote the figure: x.png\n"
    )
    assert capsys.readouterr().out.splitlines() == [
        "the script printed: first line",
        "the script printed: wrote the figure: x.png",
        "(2 indented detail lines of the script are not shown here)",
    ]


def _archive(path, entries):
    """A tar.gz of synthetic entries: (name, bytes) for a file, (name, None) for a folder, (name, 'link') for a link."""
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            elif content == "link":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/hostname"
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return path.read_bytes()


def test_an_archive_is_unpacked_once_and_only_if_every_entry_is_a_plain_file_under_its_folder(
    released, tmp_path
):
    ns, clone, cache, work = released
    good = [
        ("example", None),
        ("example/manifest.json", b"{}"),
        ("example/sub", None),
        ("example/sub/data.txt", b"numbers\n"),
    ]
    _pin(ns, clone, "example.tar.gz", _archive(tmp_path / "good.tar.gz", good))
    root = ns["unpack_archive"]("example.tar.gz", "example", "manifest.json")
    assert root == cache / "example" and (root / "sub" / "data.txt").read_bytes() == b"numbers\n"
    assert sorted(p.name for p in cache.iterdir()) == [
        "example"
    ], "a scratch folder was left in the cache"
    # a second call unpacks nothing: the folder is there, with its marker
    (root / "sub" / "data.txt").write_bytes(b"kept as it is\n")
    assert ns["unpack_archive"]("example.tar.gz", "example", "manifest.json") == root
    assert (root / "sub" / "data.txt").read_bytes() == b"kept as it is\n"
    # an unfinished earlier copy (no marker) is replaced by the archive's content
    (root / "manifest.json").unlink()
    ns["unpack_archive"]("example.tar.gz", "example", "manifest.json")
    assert (root / "sub" / "data.txt").read_bytes() == b"numbers\n"
    # entries that are not plain files under the one folder: refused, and nothing is unpacked
    for name, entries in (
        ("climbs.tar.gz", [("other", None), ("other/../../escaped.txt", b"x")]),
        ("elsewhere.tar.gz", [("other", None), ("other/ok.txt", b"x"), ("second/file.txt", b"x")]),
        ("link.tar.gz", [("other", None), ("other/link", "link")]),
        ("absolute.tar.gz", [("/other/file.txt", b"x")]),
    ):
        _pin(ns, clone, name, _archive(tmp_path / name, entries))
        with pytest.raises(RuntimeError, match="not a plain file under other"):
            ns["unpack_archive"](name, "other", "ok.txt")
        assert sorted(p.name for p in cache.iterdir()) == ["example"], name
        assert (
            not (tmp_path / "escaped.txt").exists() and not (cache.parent / "escaped.txt").exists()
        )
