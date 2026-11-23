"""The public tutorial notebook, notebooks/tutorial/hcd_priya_tutorial.ipynb, as it is committed.

The notebook is edited by hand and committed with its outputs. These are mechanical checks of the file, none of them
a rule on what the tutorial says or shows: the code cells parse and are formatted as black writes them, a function
defined in a cell is annotated and documented, and the stored outputs are those of one run from top to bottom,
without an error, a warning output or a path of a private machine.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tutorial" / "hcd_priya_tutorial.ipynb"
FORMAT_CHECK = ROOT / "tools" / "ci" / "check_notebook_format.py"
# beginnings of paths of a private machine; joined here so that this file does not hold them either
PRIVATE_PATHS = tuple("/" + name + "/" for name in ("home", "nfs", "tmp", "sw", "Users"))


def _code_cells() -> list[dict]:
    """The code cells of the committed notebook.

    Returns
    -------
    list of dict
        The cells of type "code", in order, as the notebook's JSON holds them.
    """
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return [cell for cell in cells if cell["cell_type"] == "code"]


def _as_python(cell: dict) -> str:
    """The source of a code cell with its Jupyter-only lines replaced, so that Python's parser can read it.

    Parameters
    ----------
    cell : dict
        A code cell. A line of its source that starts with ``!`` or ``%`` is a line of Jupyter, not Python.

    Returns
    -------
    str
        The source, each Jupyter-only line replaced by ``pass``. The number of lines is unchanged.
    """
    lines = "".join(cell["source"]).split("\n")
    return "\n".join("pass" if line.startswith(("!", "%")) else line for line in lines)


def _format_check() -> types.ModuleType:
    """Load the notebook format check of the CI tools from its file (tools/ci is not a package).

    Returns
    -------
    types.ModuleType
        The module ``tools/ci/check_notebook_format.py``. It imports black.
    """
    spec = importlib.util.spec_from_file_location("check_notebook_format", FORMAT_CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_code_cell_parses() -> None:
    """Each code cell is valid Python once its Jupyter-only lines are set aside."""
    for cell in _code_cells():
        try:
            ast.parse(_as_python(cell))
        except SyntaxError as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"cell {cell.get('id')} does not parse: {exc}") from None


def test_every_function_defined_in_a_cell_has_annotations_and_a_docstring() -> None:
    """A function that a cell defines says what it takes, what it returns and what it is for."""
    for cell in _code_cells():
        for node in ast.walk(ast.parse(_as_python(cell))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            where = f"cell {cell.get('id')}: {node.name}"
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            missing = [argument.arg for argument in arguments if argument.annotation is None]
            assert not missing, f"{where} has no annotation for {missing}"
            assert node.returns is not None, f"{where} has no return annotation"
            assert ast.get_docstring(node), f"{where} has no docstring"


def test_every_code_cell_is_formatted_as_black_writes_it() -> None:
    """Each code cell's source is what black gives with the repository's settings. The CI job "Format" runs the
    same check with the pinned black; here it runs where black is installed."""
    pytest.importorskip("black")
    check = _format_check()
    changed = check.cells_that_would_change(NOTEBOOK, check.black_mode())
    assert not changed, "\n\n".join(
        f"cell {name} would be reformatted:\n{diff}" for name, diff in changed
    )


def test_the_stored_outputs_are_one_run_in_order_without_an_error_or_a_private_path() -> None:
    """The committed notebook was run top to bottom, and holds no error output, no warning output and no path of
    a private machine in a cell or an output."""
    raw = NOTEBOOK.read_text(encoding="utf-8")
    for private in PRIVATE_PATHS:
        assert private not in raw, f"the committed notebook holds {private!r}"
    cells = _code_cells()
    counts = [cell["execution_count"] for cell in cells]
    assert counts == list(
        range(1, len(cells) + 1)
    ), "the outputs are not those of one run from top to bottom"
    for cell in cells:
        for output in cell["outputs"]:
            assert output["output_type"] != "error", f"cell {cell.get('id')} ended in an error"
            text = "".join(output.get("text", ""))[:200]
            assert output.get("name") != "stderr", f"cell {cell.get('id')} wrote to stderr: {text}"
