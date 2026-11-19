#!/usr/bin/env python3
"""Build one P1D held-out record (format ``hcd_priya.p1d_heldout_record/1``; hcd_analysis/emulator/p1d_heldout.py)
from explicit, digest-pinned inputs, into a NEW directory.

The record holds the held-out prediction residual of the class P1D: e = sqrt(mean(ln(prediction / truth)^2)) per
redshift, class and mode, for the LF prediction and for the LF+MF prediction, with every residual vector it was
reduced from. This script evaluates no model and computes no residual: the residuals are an input. It reduces them
(p1d_heldout.reduce_residuals), writes the record and its evidence file, reloads the record, recomputes its tables
from the evidence file and prints the record's sha256 (the digest to publish).

  python scripts/build_p1d_heldout.py --version V --out-dir NEW \\
      --residuals RESIDUALS.h5 --residuals-sha256 H --spec SPEC.json --spec-sha256 H

RESIDUALS.h5: an HDF5 file with the groups ``lf`` and ``lfmf``, each holding ``d`` (simulation, redshift, rung, 5,
mode; the five rows are clean, LLS, subDLA, DLA and the mixture at the simulation's counted class fractions; NaN: no
value), ``z`` (ascending), ``sims`` (the simulations' names, bytes), ``target_F`` (simulation, redshift, rung) and
``counts`` (simulation, redshift, rung, 4: the class sightline counts of every truth row).

SPEC.json: what only the people who measured the residuals can state. A closed object:
  {"inputs": {"lf_cache": {file, sha256}, "hr_cache": {file, sha256},
              "folds": [{fold, held_out_lf_sim, backbone_manifest_sha256, mf_record_sha256, hr_sim}, ...]},
   "code": {"commit": the commit the residuals were computed with, "dirty": false},
   "sensitivity": {"without_sim": one held-out LF simulation, "statement": why the LF table is also given without it},
   "what_it_does_not_cover": text,
   "notes": {"training_comparison", "anomalous_rows", "design_point_trend", "truth_sampling_noise": texts}}

To change a statement of the record (for example when the cause of an anomalous row is known), change SPEC.json and
build the record again into a new directory: the tables then stay the same and the record's digest changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

SPEC_KEYS = {"inputs", "code", "sensitivity", "what_it_does_not_cover", "notes"}


def parse(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", required=True, help="the record's version text")
    ap.add_argument("--out-dir", required=True, help="a NEW directory (an existing one is refused)")
    ap.add_argument("--residuals", required=True, help="the residual file (.h5)")
    ap.add_argument("--residuals-sha256", required=True)
    ap.add_argument("--spec", required=True, help="the statements and inputs (.json)")
    ap.add_argument("--spec-sha256", required=True)
    return ap.parse_args(argv)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_residuals(path):
    """{"lf": {...}, "lfmf": {...}} from a residual file, as ``p1d_heldout.write_record`` takes them."""
    import h5py
    import numpy as np

    out = {}
    with h5py.File(path, "r") as f:
        for fid in ("lf", "lfmf"):
            g = f[fid]
            out[fid] = {
                "sims": [s.decode() if isinstance(s, bytes) else str(s) for s in g["sims"][:]],
                "z": np.asarray(g["z"][:], dtype=np.float64),
                "d": np.asarray(g["d"][:], dtype=np.float32),
                "target_F": np.asarray(g["target_F"][:], dtype=np.float64),
                "counts": np.asarray(g["counts"][:], dtype=np.int64),
            }
    return out


def main(argv=None):
    a = parse(argv)
    if os.path.lexists(a.out_dir):
        sys.exit(f"refusing: {a.out_dir} exists (a record is written once into a new directory)")
    for what, path, pin in (
        ("the residual file", a.residuals, a.residuals_sha256),
        ("the spec", a.spec, a.spec_sha256),
    ):
        got = _sha256(path)
        if got != pin:
            sys.exit(f"refusing: {what} {path} has sha256 {got}, not the pinned {pin}")
    with open(a.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        sys.exit(f"refusing: the spec's keys must be exactly {sorted(SPEC_KEYS)}")
    if not isinstance(spec["sensitivity"], dict) or set(spec["sensitivity"]) != {
        "without_sim",
        "statement",
    }:
        sys.exit("refusing: spec.sensitivity must be {without_sim, statement}")

    import numpy as np

    from hcd_analysis.emulator import p1d_heldout as H
    from hcd_analysis.emulator.schema import L_BOX_HMPC

    res = read_residuals(a.residuals)
    n_modes = res["lf"]["d"].shape[-1]
    k_com = 2 * np.pi * np.arange(1, n_modes + 1) / L_BOX_HMPC
    inputs = dict(
        spec["inputs"],
        residuals={"file": os.path.basename(a.residuals), "sha256": a.residuals_sha256},
    )
    path, digest = H.write_record(
        a.out_dir,
        version=a.version,
        k_com_hmpc=k_com,
        residuals=res,
        sensitivity_without=spec["sensitivity"]["without_sim"],
        sensitivity_statement=spec["sensitivity"]["statement"],
        what_it_does_not_cover=spec["what_it_does_not_cover"],
        notes=spec["notes"],
        inputs=inputs,
        code=spec["code"],
    )
    held = H.load_record(a.out_dir, expected_record_sha256=digest)
    n = H.verify_tables(a.out_dir, held)
    print(f"record: {path}")
    print(f"record sha256 (the digest to publish): {digest}")
    print(
        f"evidence file: {held.record['tables']['bytes']} bytes, sha256 {held.record['tables']['sha256']}"
    )
    print(f"tables recomputed from {n} residual values of the evidence file: equal")
    for fid in H.FIDELITIES:
        print(
            f"{fid}: {len(held.record[fid]['sims'])} simulations, {len(held.record[fid]['z'])} redshifts"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
