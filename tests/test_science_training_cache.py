"""The CDDF against dN/dX on the LF training cache: the simulations' own counts. Release level: it runs in the assets
job, never in CI (no runner has the cache), and skips when no cache is configured.

The pipeline writes, per (simulation, snapshot), the CDDF (``cddf.measure_cddf``: a histogram of log10 N_HI over 30
bins of 0.2 dex from 17 to 23, per unit N_HI and per unit absorption distance) and the class counts
(``catalog.reconcile_counts``, each absorber classified by its own N_HI), from which the cache builder takes
dN/dX_c = count_c / total path. Both use the same path, so on counts:
- LLS: 17.2 and 19.0 are bin edges, and the CDDF summed over [17.2, 19.0) IS dN/dX_LLS;
- subDLA + DLA: the CDDF summed from 19.0 is their dN/dX less the absorbers above log N_HI = 23, which dN/dX counts
  as DLAs and the CDDF does not;
- subDLA and DLA alone: 20.3 is the centre of the bin [20.2, 20.4), so each lies between its whole-bin sum and its
  whole-bin sum plus that bin; and where that bin holds 50 or more absorbers, dN/dX puts some on each side of 20.3
  (it comes from the catalogue's N_HI values, not from the bins).

Configuration, by environment variable: HCD_SCIENCE_LF_CACHE (the path of the cache) and HCD_SCIENCE_LF_CACHE_SHA256
(its sha256, checked before anything is read; required when the path is set). Every value compared with comes from
the cache at run time; this file holds none.
"""

import hashlib
import os

import h5py
import numpy as np
import pytest

from hcd_analysis import catalog as CAT
from tests import science_cddf_sums as GC

CACHE_ENV, CACHE_SHA_ENV = "HCD_SCIENCE_LF_CACHE", "HCD_SCIENCE_LF_CACHE_SHA256"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def test_on_the_training_cache_the_cddf_and_dndx_are_the_same_counts():
    """Per (simulation, snapshot): CDDF counts (f dN path) and class counts (dN/dX path) are integers (1e-6); LLS:
    the whole-bin sum equals dN/dX (1e-12 relative); subDLA + DLA: dN/dX minus the sum from 19.0, times the path, is a
    non-negative integer; subDLA: whole <= dN/dX <= whole + the bin across 20.3; DLA: whole <= dN/dX <= whole + that
    bin + the absorbers above the last edge (1e-12 relative); where that bin holds 50 or more absorbers, both sides of
    20.3 hold some."""
    path = os.environ.get(CACHE_ENV)
    if not path:
        pytest.skip(f"no training cache configured: set {CACHE_ENV} and {CACHE_SHA_ENV}")
    want = os.environ.get(CACHE_SHA_ENV)
    assert want, f"{CACHE_SHA_ENV} must give the sha256 of the cache named by {CACHE_ENV}"
    assert _sha256(path) == want, "the cache is not the pinned file"
    with h5py.File(path, "r") as f:
        d = np.stack(
            [f["snap_dNdX_LLS"][:], f["snap_dNdX_subDLA"][:], f["snap_dNdX_DLA"][:]], axis=1
        )
        fn = f["snap_f_nhi"][:].astype(np.float64)
        e = f["log_nhi_edges"][:].astype(np.float64)
        x = f["snap_total_path_dX"][:].astype(np.float64)
    assert np.min(np.abs(e - CAT.LOG_NHI_LLS_MIN)) < GC.EDGE_TOL
    assert np.min(np.abs(e - CAT.LOG_NHI_SUBDLA_MIN)) < GC.EDGE_TOL
    assert np.min(np.abs(0.5 * (e[1:] + e[:-1]) - CAT.LOG_NHI_DLA_MIN)) < GC.EDGE_TOL
    counts = GC.per_bin_per_x(fn, e) * x[:, None]
    n_cls = d * x[:, None]
    assert np.all(np.abs(counts - np.round(counts)) < 1e-6)
    assert np.all(np.abs(n_cls - np.round(n_cls)) < 1e-6)
    assert d.shape[0] > 0
    bad = []
    for g in range(d.shape[0]):
        s = GC.class_sums(fn[g], e)
        strad = s["subDLA"]["straddle"]
        if not abs(s["LLS"]["whole"] / d[g, 0] - 1.0) <= 1e-12:
            bad.append((g, "LLS", s["LLS"]["whole"] / d[g, 0]))
        n_over = (d[g, 1] + d[g, 2] - (s["subDLA"]["whole"] + strad + s["DLA"]["whole"])) * x[g]
        if not (abs(n_over - round(n_over)) < 1e-6 and round(n_over) >= 0):
            bad.append((g, "subDLA+DLA", n_over))
        for c, j, extra in (("subDLA", 1, 0.0), ("DLA", 2, max(round(n_over), 0) / x[g])):
            lo, hi = s[c]["whole"], s[c]["whole"] + strad + extra
            if not (lo <= d[g, j] * (1 + 1e-12) and d[g, j] <= hi * (1 + 1e-12)):
                bad.append((g, c, lo, d[g, j], hi))
        # a dN/dX taken from binned counts (the bin across 20.3 given whole to one class, as an early cache builder
        # did) puts every absorber of that bin on one side; with 50 or more of them, both sides are populated
        n_strad, n_below = strad * x[g], (d[g, 1] - s["subDLA"]["whole"]) * x[g]
        if n_strad >= 50 and not (0.5 <= n_below <= n_strad - 0.5):
            bad.append((g, "the bin across 20.3 all on one side", n_below, n_strad))
    assert not bad, bad[:10]
