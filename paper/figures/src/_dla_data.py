"""Measured absorber statistics from the public DLA data collection, read as its own reader reads them.

The collection: github.com/jibanCat/DLA_data (a fork of github.com/sbird/DLA_data), at the commit below.
It supplies the DLA, subDLA and LLS points of the CDDF figure of the paper (Figure 4).
Each file is checked by sha256 before it is read. Every function returns linear quantities: N_HI in cm^-2,
f(N_HI) in cm^2, and the lower and upper ends of the interval the source gives (never a symmetric error).

What each file is, from the docstrings of the collection's reader `dla_data.py` (not checked against the papers):
  ho21/dndx_all.txt     DLA dN/dX, Ho, Bird and Garnett 2021 (rows confirmed by its first author)
  ho21/cddf_z*.txt      DLA f(N_HI) of the same paper in redshift bins 2-2.5, 2.5-3, 3-4
  zafar_2013.txt        subDLA f(N_HI), Zafar et al. 2013 (arXiv:1307.0602), z 1.5 to 5
  summary.dat           f(N_HI) for 19.0 < log N_HI < 20.3, O'Meara et al. 2007, mean z 3.1
  prochaska_lls.dat     LLS f(N_HI) as a region (a polygon), Prochaska, O'Meara and Worseck 2010, z 3.7
No file of the collection gives the incidence dN/dX of LLS or of subDLAs.
"""

import hashlib

import numpy as np

from _public import fetch

REPO = "jibanCat/DLA_data"
COMMIT = "daa595c692254c93b86009318a6e2b029db67112"
SHA256 = {
    "ho21/dndx_all.txt": "fb4d86151b34240902dbd6e63914e7c98dbb367fb6880a0e6871c849345ca81c",
    "ho21/cddf_z225.txt": "feaffb7b88215e767ca53a41fb3f00b65c90aaf27745c28720b32976a9168a63",
    "ho21/cddf_z253.txt": "62e93a1e1977b3d78b9f56475bd4390c1eee7eaf371986ca5ece2db338bb7cc7",
    "ho21/cddf_z34.txt": "f21dca06cd5974a08de13632ef74109f5b78afc2921f86ec9b3c6cf9f838ad3a",
    "zafar_2013.txt": "f592ecfd74bb423eae15d9bf40c7aca3e3da61ee7bbc1dab9510e6edca4767bd",
    "summary.dat": "19a3e15287d9f4d053fba2686ad985a51fba47c1a5b8963bee175b55f5bea40d",
    "prochaska_lls.dat": "e0dd4bad98e5396acad183fb88b29562adcd83fd3d9b80c39a431a1c172c319e",
}


class Missing(Exception):
    pass


def table(path):
    data = fetch(COMMIT, path, repo=REPO)
    if data is None:
        raise Missing(path)
    got = hashlib.sha256(data).hexdigest()
    want = SHA256[path]
    if want is not None and got != want:
        raise SystemExit(f"{path} of {REPO} at {COMMIT[:7]} has sha256 {got}, expected {want}")
    lines = [ln for ln in data.decode().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    return np.array([[float(x) for x in ln.split()] for ln in lines])


def ho21_dndx():
    """z, dN/dX, lower and upper end of the 68 per cent interval (rows 0 to 3; DLAs only)."""
    d = table("ho21/dndx_all.txt")
    return d[0], d[1], d[2], d[3]


HO21_CDDF_FILE = {"2-2.5": "ho21/cddf_z225.txt", "2.5-3": "ho21/cddf_z253.txt", "3-4": "ho21/cddf_z34.txt"}


def ho21_cddf(zbin):
    """N_HI, f, lower and upper end of the 68 per cent interval; bins with f = 0 are left out (no detection)."""
    d = table(HO21_CDDF_FILE[zbin])
    keep = d[1] > 0
    return 10 ** d[0][keep], d[1][keep], d[2][keep], d[3][keep]


def zafar13():
    """As `zafar_data`: columns log N, half-width in log N, log f, upper and lower error of log f."""
    d = table("zafar_2013.txt")
    n, f = 10 ** d[:, 0], 10 ** d[:, 2]
    return n, f, 10 ** (d[:, 2] - d[:, 4]), 10 ** (d[:, 2] + d[:, 3]), 10 ** (d[:, 0] - d[:, 1]), 10 ** (d[:, 0] + d[:, 1])


def omeara07():
    """As `omeara_data` (format_error_bars): columns log N, log f, log N min, log N max, log f min, log f max."""
    d = 10 ** table("summary.dat")
    return d[:, 0], d[:, 1], d[:, 4], d[:, 5], d[:, 2], d[:, 3]


def prochaska10():
    """As `prochaska_10_data`: the corners of a region in (log N, log f), drawn filled."""
    d = table("prochaska_lls.dat")
    return 10 ** d[:, 0], 10 ** d[:, 1]
