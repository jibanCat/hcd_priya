# The public tutorial

`hcd_priya_tutorial.ipynb` is one notebook in three parts: the scientific motivation, the emulator, and a mock
inference. It runs from an installed package and needs no file of the PRIYA simulations.

**PROVISIONAL (PI to refine).**

## How to run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "hcd-priya[tutorials]"
jupyter notebook hcd_priya_tutorial.ipynb
```

The trained model comes with the package. The notebook file alone is enough: the few stored files it reads are
downloaded once into `~/.cache/hcd_priya_tutorial` (or the folder named by the environment variable
`HCD_PRIYA_TUTORIAL_CACHE`). Inside a clone of the repository the files on disk are used instead. In both cases a
file is used only when its sha256 is the one written in the notebook.

## The files here

| File | What it is |
|---|---|
| `hcd_priya_tutorial.ipynb` | the notebook, committed with its outputs |
| `_build_tutorial.py` | the builder: every cell as source; `python _build_tutorial.py` writes the notebook without outputs |
| `assets/ASSETS.json` | every stored file the notebook reads: path, sha256, size, URL, what it is, where it came from |
| `assets/figures/` | the schematic of the emulator shown in Part II, with `PROVENANCE.md` |
| `run_mock_chain.py` | the preflight and the one registered run of the stored mock chains of Part III; it runs the notebook's own cells |

## What is pending

The stored chains of Part III have not been run. They are run once, after the project lead's approval of the
preflight. Until then the cells that read them, the contours and the posterior predictive are text cells marked
PENDING, and no chain file is in this folder. The image of Fig. 1 is a placeholder.

## Where the stored files come from

Every URL in `assets/ASSETS.json` is immutable: the tag `v1.0.0` of this repository (the literature tables under
`docs/figures/data/`, each folder with its own `PROVENANCE.md`), a file of the GitHub release `v1.0.0` (the
example data set), or one commit of this repository (files added for the tutorial). No file is fetched from a
branch.
