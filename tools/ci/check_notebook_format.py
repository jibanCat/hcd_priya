"""Check that the code cells of a Jupyter notebook are formatted as black writes them.

black reads Python files only (pyproject.toml, [tool.black]). This script applies the same black, with the line
length and the target version of pyproject.toml, to the source of each code cell of a notebook, one cell at a time.
It changes no file: it prints what would change in each cell and exits with 1 if any cell would change. A line that
starts with ``!`` or ``%`` is Jupyter's own, not Python, and is left as it is.

    python tools/ci/check_notebook_format.py notebooks/tutorial/hcd_priya_tutorial.ipynb

It needs black (tools/ci/requirements-format.txt) and the standard library. The CI job "Format" runs it on the
tutorial notebook; no test of the test suite calls it, because the test environment does not hold black.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import sys
import tomllib

import black

ROOT = pathlib.Path(__file__).resolve().parents[2]
JUPYTER_LINE = "pass  # a line of Jupyter, not Python"


def black_mode(pyproject: pathlib.Path = ROOT / "pyproject.toml") -> black.Mode:
    """The black settings of the repository.

    Parameters
    ----------
    pyproject : pathlib.Path
        The ``pyproject.toml`` whose table ``[tool.black]`` gives the line length and the target versions.

    Returns
    -------
    black.Mode
        The mode that ``black .`` uses in this repository.
    """
    settings = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["black"]
    versions = {black.TargetVersion[name.upper()] for name in settings["target-version"]}
    return black.Mode(line_length=settings["line-length"], target_versions=versions)


def cells_that_would_change(notebook: pathlib.Path, mode: black.Mode) -> list[tuple[str, str]]:
    """The code cells of a notebook that black would reformat.

    Parameters
    ----------
    notebook : pathlib.Path
        A notebook in the JSON format, version 4.
    mode : black.Mode
        The black settings to format with.

    Returns
    -------
    list of (str, str)
        Per cell that would change: its id (or its position, for a cell without an id) and a unified diff from
        its source to what black writes. Empty when every code cell is formatted as black writes it.
    """
    changed = []
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for position, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # black cannot parse a shell line or a magic, so each is replaced by a statement that black leaves alone
        lines = [
            JUPYTER_LINE if line.startswith(("!", "%")) else line for line in source.split("\n")
        ]
        python = "\n".join(lines)
        formatted = black.format_str(python, mode=mode).rstrip("\n")
        if formatted != python:
            name = str(cell.get("id", f"cell {position}"))
            diff = difflib.unified_diff(
                python.split("\n"), formatted.split("\n"), "committed", "black", lineterm=""
            )
            changed.append((name, "\n".join(diff)))
    return changed


def main(argv: list[str]) -> int:
    """Check each notebook named on the command line and report.

    Parameters
    ----------
    argv : list of str
        The paths of the notebooks to check.

    Returns
    -------
    int
        0 when no code cell of any notebook would change, 1 otherwise.
    """
    if not argv:
        print("usage: check_notebook_format.py NOTEBOOK [NOTEBOOK ...]")
        return 2
    mode = black_mode()
    would_change = 0
    for path in argv:
        changed = cells_that_would_change(pathlib.Path(path), mode)
        would_change += len(changed)
        for name, diff in changed:
            print(f"{path}: the code cell {name} would be reformatted:\n{diff}\n")
        if not changed:
            print(f"{path}: every code cell is formatted as black writes it")
    return 1 if would_change else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
