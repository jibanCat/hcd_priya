"""The CDDF summed per HCD class, for the science tests of the CDDF (test_science_cddf.py, test_science_model.py,
test_science_training_cache.py).

Pure numpy, no model and no number of any model. The class ranges are the package's catalogue thresholds
(``hcd_analysis.catalog.LOG_NHI_*_MIN``: LLS [17.2, 19.0), subDLA [19.0, 20.3), DLA >= 20.3, log10 N_HI in cm^-2).

The CDDF is f(N_HI) = d^2 n / (dN_HI dX): absorbers per unit column density (linear N_HI, cm^-2) per unit absorption
distance X (``hcd_analysis.cddf``; ``Prediction.cddf``). So f_i x (10^upper_i - 10^lower_i) is the number of absorbers
per unit X in bin i, and its sum over a class's bins is that class's dN/dX, provided every bin lies wholly inside the
class. A bin across a class edge cannot be split from f alone: how its absorbers fall on the two sides of the edge
is not in the CDDF. ``class_sums`` therefore returns, per class, the sum over the WHOLE bins only, and separately the
content of each bin that straddles an edge of that class: the class's dN/dX lies between the whole-bin sum and the
whole-bin sum plus the straddling content, for any distribution inside the bin.
"""

import numpy as np

from hcd_analysis import catalog as CAT

EDGE_TOL = 1e-9  # in log10 N_HI: a bin edge "on" a class edge
CLASS_RANGES = {
    "LLS": (CAT.LOG_NHI_LLS_MIN, CAT.LOG_NHI_SUBDLA_MIN),
    "subDLA": (CAT.LOG_NHI_SUBDLA_MIN, CAT.LOG_NHI_DLA_MIN),
    "DLA": (CAT.LOG_NHI_DLA_MIN, np.inf),
}


def edges_from_centres(centres):
    """Bin edges from uniformly spaced bin centres (the centres +- half their spacing)."""
    c = np.asarray(centres, dtype=np.float64)
    step = np.diff(c)
    if c.size < 2 or not np.allclose(step, step[0], rtol=1e-9, atol=0):
        raise ValueError("the bin centres are not uniformly spaced")
    return np.concatenate([c - 0.5 * step[0], [c[-1] + 0.5 * step[0]]])


def per_bin_per_x(f, edges):
    """Absorbers per unit X in each bin: f x (10^upper - 10^lower). Any leading axes of f are kept."""
    e = np.asarray(edges, dtype=np.float64)
    return np.asarray(f, dtype=np.float64) * (10.0 ** e[1:] - 10.0 ** e[:-1])


def class_sums(f, edges):
    """Per class: {"whole": sum over the bins wholly inside the class, "straddle": sum over the bins across an edge of
    the class, "whole_bins": their indices, "straddle_bins": their indices, "nan_bins": indices of NaN bins among both
    (counted as zero)}. f has shape (Nbin,)."""
    e = np.asarray(edges, dtype=np.float64)
    lo, hi = e[:-1], e[1:]
    s = per_bin_per_x(f, e)
    out = {}
    for c, (a, b) in CLASS_RANGES.items():
        whole = (lo >= a - EDGE_TOL) & (hi <= b + EDGE_TOL)
        straddle = ~whole & (hi > a + EDGE_TOL) & (lo < b - EDGE_TOL)
        nan = ~np.isfinite(s) & (whole | straddle)
        out[c] = {
            "whole": float(np.sum(np.where(whole & np.isfinite(s), s, 0.0))),
            "straddle": float(np.sum(np.where(straddle & np.isfinite(s), s, 0.0))),
            "whole_bins": np.where(whole)[0],
            "straddle_bins": np.where(straddle)[0],
            "nan_bins": np.where(nan)[0],
        }
    return out
