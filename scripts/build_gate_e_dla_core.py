#!/usr/bin/env python3
"""Gate E amendment A1 rev 1 section 4: the DLA-core product. Per production leg (DESI, KS k <= 0.065, eBOSS) one core,
a fixed function of physical velocity-space k: the mean over the leg's z of the per-z mean over the cache rows at that
z (|z - z_grid| < 0.05, all simulations and rungs) of each row's DLA core (cache delta[:, 2]) interpolated linearly in k
from the row's OWN stored grid (kfkms) onto a fixed log grid (400 points, 4e-4 to 0.1 s/km). The forward reads it at
the data bins by linear interpolation in ln k (no binding, no theta-dependence). Write-once product kind ``dla_core``.

ENV: PYTHONNOUSERSITE=1 PYTHONPATH=<repo> JAX_PLATFORMS=cpu /home/mfho/.conda/envs/emu-jax/bin/python3 \
     scripts/build_gate_e_dla_core.py --cache <lf cache> --out <product.npz>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np

from hcd_analysis.emulator import cemu_build as CB
from hcd_analysis.emulator import data_likelihood as DL
from hcd_analysis.emulator.data import load_cache
from hcd_analysis.emulator.products import save_product
from hcd_analysis.emulator.schema import L_BOX_HMPC

ROW_RULE = ("per leg z: cache rows with |z_grid - z| < 0.05 (all simulations and rungs); each row's delta[:, 2] "
            "interpolated linearly in k at fixed physical k from its own kfkms grid (finite modes; rows not covering a "
            "point excluded there); mean over rows; mean over the leg's z; fixed log grid 400 points [4e-4, 0.1] s/km; "
            "read at data bins by linear interpolation in ln k")


def legs():
    return [DL.load_desi_leg(metals_on=True, resolution_float=True, resolution_coherent=False),
            DL.load_ks_leg(resolution_float=True, k_max=0.065),
            DL.load_eboss_leg(resolution_float=True, resolution_coherent=False)]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    d = load_cache(a.cache)
    k_com = 2 * np.pi * np.arange(1, 173) / L_BOX_HMPC
    np.testing.assert_allclose(np.asarray(d["k_com_hmpc"]), k_com, rtol=1e-13)
    kf = np.asarray(d["kfkms"], float)
    if not np.all(np.diff(kf, axis=1) > 0):
        raise SystemExit("cache kfkms not increasing along modes")
    core = np.asarray(d["delta"], float)[:, 2]
    z_rows = np.asarray(d["z_grid"], float)
    arrays, leg_z = {"k_grid": CB.DLA_CORE_GRID}, {}
    for leg in legs():
        zl = [float(z) for z in np.asarray(leg.z)]
        c = CB.dla_core_leg(kf, core, z_rows, zl, CB.DLA_CORE_GRID)
        keep = np.isfinite(np.asarray(leg.P_data))
        at = CB.dla_core_at(np.asarray(leg.k)[keep], CB.DLA_CORE_GRID, c)        # refuses uncovered data k
        if not np.all(np.isfinite(at)):
            raise SystemExit(f"{leg.name}: non-finite core at a data bin")
        arrays[f"core_{leg.name}"] = c
        leg_z[leg.name] = zl
        print(f"{leg.name}: z {zl[0]:.1f}-{zl[-1]:.1f} ({len(zl)}), finite grid points {int(np.isfinite(c).sum())}/400, "
              f"kept bins {int(keep.sum())} covered")
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    prov = dict(code_commit=commit, cache_sha256=_sha(a.cache), inputs={"cache": a.cache}, row_rule=ROW_RULE,
                leg_z=leg_z, amendment="GATE_E_AMENDMENT_A1 rev 1 section 4 (PU-0068)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_product(a.out, "dla_core", k_com_hmpc=k_com, provenance=prov, **arrays)
    print(f"wrote {a.out} sha256 {_sha(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
