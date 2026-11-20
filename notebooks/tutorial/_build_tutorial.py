"""Build the public tutorial notebook, ``notebooks/tutorial/hcd_priya_tutorial.ipynb``.

Every cell of the notebook is held here as source, in order, in ``CELLS``: a list of ``(kind, cell_id, source)``
with kind ``"markdown"`` or ``"code"``. ``build(path)`` writes them as a notebook without outputs; the committed
notebook is that file, executed. The helper functions the notebook needs are cells of the notebook, so the
notebook imports nothing beyond the installed packages; ``helper_source(cell_id)`` returns the source of one code
cell, which is how the tests (``tests/test_tutorial_notebook.py``) and ``run_mock_chain.py`` run the same code a
reader runs.

Run::

    python notebooks/tutorial/_build_tutorial.py            # writes the notebook beside this file
    jupyter nbconvert --to notebook --execute --inplace notebooks/tutorial/hcd_priya_tutorial.ipynb

The table of stored files in the cell ``helpers-assets`` is written from ``assets/ASSETS.json``.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
NOTEBOOK = HERE / "hcd_priya_tutorial.ipynb"
MANIFEST = HERE / "assets" / "ASSETS.json"

CELLS: list = []


def text(cell_id: str, source: str) -> None:
    """Add a text cell."""
    CELLS.append(("markdown", cell_id, source.strip("\n")))


def code(cell_id: str, source: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", cell_id, source.strip("\n")))


def helper_source(cell_id: str) -> str:
    """The source of the code cell ``cell_id``, as it stands in the notebook.

    Raises
    ------
    KeyError
        If no code cell has that id.
    """
    for kind, name, source in CELLS:
        if name == cell_id and kind == "code":
            return source
    raise KeyError(cell_id)


def assets_literal() -> str:
    """The table ``ASSETS`` of the cell ``helpers-assets``, as Python source, from ``assets/ASSETS.json``."""
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    rows = [
        f'    "{r["path"]}": {{\n        "sha256": "{r["sha256"]}",\n        "url": "{r["url"]}",\n    }},'
        for r in records
    ]
    return "ASSETS = {\n" + "\n".join(rows) + "\n}"


def build(out_path=NOTEBOOK) -> pathlib.Path:
    """Write the notebook, without outputs, to ``out_path``."""
    cells = []
    for kind, cell_id, source in CELLS:
        cell = {
            "cell_type": kind,
            "id": cell_id,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    doc = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = pathlib.Path(out_path)
    out_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


# ============================================================================================ the cells
text(
    "title",
    r"""
# From HCD absorbers to a mock inference: a tutorial for `hcd_priya`

**PROVISIONAL (PI to refine).** This notebook has three parts.

1. **Scientific motivation.** The Lyman-alpha forest and the high-column-density (HCD) absorbers on the same
   sightlines: Lyman limit systems (LLS), sub-damped and damped Lyman-alpha absorbers (subDLAs, DLAs).
2. **The emulator.** What the published model `hcd_priya` 1.0.0 returns: the flux power spectrum (P1D) of four
   classes of sightline, the incidence of the absorbers, and a mixture of the classes.
3. **Mock inference.** A Gaussian likelihood on a mock made by the model itself, with four free parameters. It is
   a self-consistency demonstration: not an independent validation, not a result from data.

**To run it:** `pip install "hcd-priya[tutorials]"` in Python 3.11 or later, then open this file with Jupyter. The
trained model comes with the package. The few stored files the notebook reads are found in a clone of the
repository or downloaded once, each checked against its sha256 (next cell).
""",
)

code(
    "imports",
    r"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

import hcd_analysis
import hcd_analysis.predictor as hp

print("hcd_analysis", hcd_analysis.__version__)
""",
)

code(
    "helpers-assets",
    r"""
# The stored files of this tutorial. Each is named by its path in the repository (a file of the GitHub release by
# its file name) and pinned by its sha256. Every URL is immutable: the tag v1.0.0, a file of the release v1.0.0,
# or one commit.
import hashlib
import os
import pathlib
import tempfile
import urllib.request

@ASSETS@

# Inside a clone of the repository the files are on disk already. A local file is used when, and only when, its
# sha256 is the pinned one, so the notebook reads the same bytes inside a clone and outside one. This notebook's
# folder is two levels below the top of a clone.
LOCAL_ROOTS = (pathlib.Path.cwd(), pathlib.Path.cwd().parent.parent)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tutorial_file(relpath):
    # The path of a stored file of the tutorial, with the pinned sha256: a local copy in a clone, a copy
    # downloaded earlier, or a new download into the cache folder (the environment variable
    # HCD_PRIYA_TUTORIAL_CACHE, default ~/.cache/hcd_priya_tutorial).
    if relpath not in ASSETS:
        raise RuntimeError(f"{relpath} is not a stored file of this tutorial (ASSETS names {len(ASSETS)} files)")
    pinned, url = ASSETS[relpath]["sha256"], ASSETS[relpath]["url"]
    for root in LOCAL_ROOTS:
        local = pathlib.Path(root) / relpath
        if local.is_file() and sha256_of(local) == pinned:
            return local
    cache = pathlib.Path(os.environ.get("HCD_PRIYA_TUTORIAL_CACHE", "~/.cache/hcd_priya_tutorial")).expanduser()
    target = cache / relpath
    if target.is_file() and sha256_of(target) == pinned:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # written under a temporary name and renamed only after the digest is checked: no partial file is left
    handle, partial = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
    digest = hashlib.sha256()
    try:
        try:
            with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
                    digest.update(block)
        except Exception as exc:
            raise RuntimeError(
                f"could not download the tutorial file {relpath} from {url} into the cache folder {cache}: {exc}"
            ) from exc
        if digest.hexdigest() != pinned:
            raise RuntimeError(
                f"the tutorial file {relpath} downloaded from {url} has sha256 {digest.hexdigest()}, "
                f"not the pinned {pinned}: it is refused, and nothing is kept in {cache}"
            )
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return target


print(len(ASSETS), "stored files are pinned by sha256")
""".replace("@ASSETS@", assets_literal()),
)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = build(argv[0] if argv else NOTEBOOK)
    print(f"{out}: {len(CELLS)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
