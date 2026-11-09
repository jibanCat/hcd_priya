"""Raw-simulation header reader shared by gate tests: extracted VERBATIM from 863df63 tests/test_parity_upstream.py
(lines 490-505) so the gate D review test can use it without importing the upstream-parity module, which executes the
upstream repository (excluded from PR-0). It reads HDF5 headers only."""
import os

import h5py
import numpy as np


def _raw_headers(raw_root, sim):
    """Header facts of every raw grid_480 savefile of one simulation; vmax exactly as fake_spectra (spectra.py:208-215)."""
    import glob
    out = []
    for d in sorted(glob.glob(f"{raw_root}/{sim}/output/SPECTRA_*")):
        p = f"{d}/lya_forest_spectra_grid_480.hdf5"
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            a = f["Header"].attrs
            z, h = float(a["redshift"]), float(a["hubble"])
            at = 1.0 / (1.0 + z)
            hz = float(a["Hz"]) if "Hz" in a else 100.0 * h * np.sqrt(float(a["omegam"]) / at ** 3 + float(a["omegal"]))
            vmax = float(a["box"]) * (3.085678e21 * at / h) * hz / 3.085678e24
            out.append(dict(spec=int(d.rsplit("_", 1)[1]), z=z, vmax=vmax, nbins=int(a["nbins"])))
    return out
