# The public tutorial

`hcd_priya_tutorial.ipynb` is one notebook in three parts: the scientific motivation, the emulator, and the public
likelihood with the repository's mock-recovery example. It teaches the public functions of the package. It adds no
method of its own: it calls the package, three released figure scripts and the released example.

**UNAPPROVED (PI to review).**

## How to run it

From a clone of the repository:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "hcd-priya[tutorials]"
cd notebooks/tutorial
curl -L -O https://github.com/jibanCat/hcd_priya/releases/download/v1.0.0/hcd_priya_example_download.tar.gz
tar -xzf hcd_priya_example_download.tar.gz
jupyter notebook hcd_priya_tutorial.ipynb
```

The trained model comes with the package. The notebook needs the clone: three of its figures are drawn by the
scripts under `scripts/figures/`, and Part III uses `examples/mock_recovery.py` with the example data set of the
release, unpacked beside the notebook as above. The example checks each file of that data set when it opens it.

## The files here

| File | What it is |
|---|---|
| `hcd_priya_tutorial.ipynb` | the notebook, committed with its outputs |
| `_build_tutorial.py` | the builder: every cell as source; `python _build_tutorial.py` writes the notebook without outputs |
| `assets/figures/` | the schematic of the emulator shown in Part II, with `PROVENANCE.md` |

The figures that the released code writes while the notebook runs go into `tutorial_output/` beside the notebook.
Git ignores that folder and the example data set.

## What is pending

The image of Fig. 1 is a placeholder. No text cell is approved.
