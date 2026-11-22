"""The public tutorial notebook, notebooks/tutorial/: its builder, its cells and the committed, executed notebook.

The notebook is written by ``notebooks/tutorial/_build_tutorial.py``, which holds every cell as source. It is a
beginner's tutorial of the public functions of the package: its code cells are short, define no helper of their
own for files or downloads, and call the installed package, the released figure scripts of the clone and the
released example. These tests keep it that way. Every value in them is synthetic.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "notebooks" / "tutorial"
BUILDER = TUTORIAL / "_build_tutorial.py"
NOTEBOOK = TUTORIAL / "hcd_priya_tutorial.ipynb"

# strings of a private machine that no public text holds; joined here so that this file does not hold them either
PRIVATE_PATHS = tuple("/" + name + "/" for name in ("home", "nfs", "tmp")) + ("tur" + "bo",)
# a code cell is one step a new reader can take in at once
LONGEST_CODE_CELL = 25
# the notebook is not the place for the release's file checks: no digest, no download code
FORBIDDEN_WORDS = ("sha" + "256", "url" + "lib", "hash" + "lib")
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


def _builder() -> types.ModuleType:
    """Load the notebook's builder from its file.

    Returns
    -------
    types.ModuleType
        The module ``notebooks/tutorial/_build_tutorial.py``, freshly executed, with ``CELLS`` and ``build``.
    """
    spec = importlib.util.spec_from_file_location("_build_tutorial", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_cells(path: pathlib.Path) -> list[tuple[str, str, str]]:
    """Read the cells of a notebook file.

    Parameters
    ----------
    path : pathlib.Path
        A notebook in the JSON format, version 4.

    Returns
    -------
    list of (str, str, str)
        Per cell, in order: its type ("markdown" or "code"), its id and its source as one string.
    """
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [(c["cell_type"], c["id"], "".join(c["source"])) for c in doc["cells"]]


def _as_python(source: str) -> str:
    """The source of a code cell with its shell lines replaced, so that Python tools can read it.

    Parameters
    ----------
    source : str
        The source of a code cell. A line that starts with ``!`` is a shell line of Jupyter, not Python.

    Returns
    -------
    str
        The same lines, each shell line replaced by ``pass`` with a comment. The number of lines is unchanged.
    """
    lines = [
        "pass  # a shell line" if line.startswith("!") else line for line in source.split("\n")
    ]
    return "\n".join(lines)


def _code_cells() -> list[tuple[str, str]]:
    """The code cells of the builder.

    Returns
    -------
    list of (str, str)
        Per code cell, in order: its id and its source.
    """
    return [(cell_id, source) for kind, cell_id, source in _builder().CELLS if kind == "code"]


# ------------------------------------------------------------------------------------------ the builder
def test_builder_output_equals_the_committed_cells(tmp_path: pathlib.Path) -> None:
    """The committed notebook holds exactly the builder's cells; a built notebook has no output."""
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
    code = [c for c in doc["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)


def test_every_code_cell_parses() -> None:
    """Each code cell is valid Python once its shell lines are set aside."""
    for cell_id, source in _code_cells():
        try:
            ast.parse(_as_python(source))
        except SyntaxError as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"cell {cell_id} does not parse: {exc}") from None


def test_no_private_attribute_no_private_path_no_em_dash() -> None:
    """No cell and no file of the tutorial holds a private path or an em-dash, and no cell uses a private name."""
    texts = {f"cell {cell_id}": source for _, cell_id, source in _builder().CELLS}
    for path in (BUILDER, TUTORIAL / "README.md"):
        texts[path.name] = path.read_text(encoding="utf-8")
    for name, text in texts.items():
        assert "—" not in text, f"{name} holds an em-dash"
        for private in PRIVATE_PATHS:
            assert private not in text, f"{name} holds {private!r}"
    for cell_id, source in _code_cells():
        for node in ast.walk(ast.parse(_as_python(source))):
            names = []
            if isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split(".")[-1] for a in node.names]
            for n in names:
                private = n.startswith("_") and not (n.startswith("__") and n.endswith("__"))
                assert not private, f"cell {cell_id} uses the private name {n}"


# ------------------------------------------------------------------------------------------ a beginner's notebook
def test_every_code_cell_is_short() -> None:
    """No code cell is longer than a new reader can take in at once."""
    for cell_id, source in _code_cells():
        lines = source.split("\n")
        assert len(lines) <= LONGEST_CODE_CELL, f"cell {cell_id} has {len(lines)} lines"


def test_no_cell_holds_file_checking_or_download_code() -> None:
    """The release's file checks stay in the project's own code: no cell names a digest or a download library, and
    the tutorial's folder holds no manifest of stored files and no script beside the builder."""
    for _, cell_id, source in _builder().CELLS:
        for word in FORBIDDEN_WORDS:
            assert word not in source.lower(), f"cell {cell_id} holds {word!r}"
    scripts = sorted(p.name for p in TUTORIAL.glob("*.py"))
    assert scripts == ["_build_tutorial.py"], "a second script beside the builder"
    assert not (
        TUTORIAL / "assets" / "ASSETS.json"
    ).exists(), "a manifest of stored files in the tutorial's folder"


def test_the_notebook_defines_no_inference_code_of_its_own() -> None:
    """The notebook calls the package, the released figure scripts and the released example. It defines no sampler,
    no posterior and no chain code, and imports no sampler or optimiser."""
    for cell_id, source in _code_cells():
        for node in ast.walk(ast.parse(_as_python(source))):
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
                refused = any(
                    module == bad or module.startswith(bad + ".") for bad in FORBIDDEN_IMPORTS
                )
                assert not refused, f"cell {cell_id} imports {module}"


def test_every_function_of_a_cell_and_of_the_builder_has_annotations_and_a_docstring() -> None:
    """A function that tutorial-owned code defines says what it takes, what it returns and what it is for."""
    sources = {f"cell {cell_id}": _as_python(source) for cell_id, source in _code_cells()}
    sources["the builder"] = BUILDER.read_text(encoding="utf-8")
    sources["this test file"] = pathlib.Path(__file__).read_text(encoding="utf-8")
    checked = 0
    for where, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            checked += 1
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            missing = [a.arg for a in arguments if a.annotation is None]
            assert not missing, f"{where}: {node.name} has no annotation for {missing}"
            assert node.returns is not None, f"{where}: {node.name} has no return annotation"
            assert ast.get_docstring(node), f"{where}: {node.name} has no docstring"
    assert checked >= 4  # the builder's own functions, at the least


def test_every_code_cell_is_formatted_as_black_writes_it() -> None:
    """Each code cell's source is what Black gives at the repository's line length (a shell line set aside)."""
    black = pytest.importorskip("black")
    for cell_id, source in _code_cells():
        python = _as_python(source)
        formatted = black.format_str(python, mode=black.Mode(line_length=100)).rstrip("\n")
        assert formatted == python, f"cell {cell_id} is not formatted as Black writes it"


# ------------------------------------------------------------------------------------------ the committed notebook
def test_the_committed_notebook_is_executed_in_order_without_an_error_or_a_private_path() -> None:
    """The committed notebook was run top to bottom, and holds no error, no warning output and no private path."""
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
            stderr = output.get("name") == "stderr"
            assert (
                not stderr
            ), f"cell {c['id']} wrote to stderr: {''.join(output.get('text', ''))[:200]}"
