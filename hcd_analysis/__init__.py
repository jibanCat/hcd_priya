"""
hcd_analysis: the per-class P1D emulator of high-column-density (HCD) absorbers on the PRIYA simulations, its
Gaussian likelihood, and the absorber pipeline it was trained from.

The public interface:
  predictor.py       load a model (load_published, load_release) and predict per-class P1D, w_c, dN/dX and the CDDF
  likelihood.py      the Gaussian P1D likelihood and the reading and writing of data sets and references
  check_release.py   checking an installed model release from a fresh install
  published_models.py  the model releases this version accepts, with their digests

The pipeline and the research code:
  config.py, io.py, snapshot_map.py  configuration, HDF5 reading, the snapshot-to-redshift map
  catalog.py, masking.py             absorber identification from tau (LLS, subDLA, DLA) and masking
  p1d.py, cddf.py                    the P1D and the CDDF of the simulations
  pipeline.py, report.py             orchestration, figures and markdown outputs
  emulator/, comparison/             the emulator's training and research inference; emulator comparisons
"""

__version__ = "0.2.0"
