# The public tutorial

`hcd_priya_tutorial.ipynb` is one notebook in three parts: the scientific motivation, the emulator, and the public
likelihood with the repository's mock-recovery example. It teaches the public functions of the package in plain Python. It adds no
method of its own, and it runs no script: every figure is drawn in the notebook from what the package returns.

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

The trained model comes with the package. The notebook needs the clone: three of its figures read published
measurements and a table of the training simulations from `docs/figures/data/`, and Part III reads the list of the
example's files from `examples/mock_recovery.py` and the example data set of the release, unpacked beside the
notebook as above. The package's loaders check each file of that data set when they read it.

## The files here

| File | What it is |
|---|---|
| `hcd_priya_tutorial.ipynb` | the notebook, committed with its outputs. It is the file to edit: after an edit, run it top to bottom and commit it with its outputs |
| `assets/figures/` | the schematic of the emulator shown in Part II, with `PROVENANCE.md` |

Git ignores the example data set unpacked beside the notebook.
