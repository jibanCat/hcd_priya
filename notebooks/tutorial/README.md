# Tutorial for `hcd_priya`

In this tutorial, we will go through the scientific motivation behind `hcd_priya` and how to use the package.

We use the public functions of `hcd_priya` throughout the notebook, so you can use the code here as a starting point for your own analysis.

## How to run the tutorial

From a clone of the repository:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "hcd-priya[tutorials]"
cd notebooks/tutorial
curl -L -O https://github.com/jibanCat/hcd_priya/releases/download/v1.0.0/hcd_priya_example_download.tar.gz
tar -xzf hcd_priya_example_download.tar.gz
jupyter notebook hcd_priya_tutorial.ipynb
```

The trained emulator comes with the package. We use some published measurements and simulation data available in `docs/figures/data/` for comparison. For the mock-recovery analysis in Part III, we use the example data set from the v1.0.0 release, which can be downloaded and unpacked using the commands above.

## Files

| File | Description |
|---|---|
| `hcd_priya_tutorial.ipynb` | The tutorial notebook, including its outputs. |
| `assets/figures/` | The schematic of the HCD emulator used in Part II and its provenance. |

The downloaded example data set is excluded from Git.

