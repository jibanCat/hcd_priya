"""Cross-version check of the HDF5 release format (hcd_priya.h5/2): a bundle written with one libhdf5 is read, bit for
bit, with another (the release writer keeps files readable by libhdf5 1.10).

usage:
  python tools/ci/h5_cross_version.py write NEW_DIR      # with the pinned stack: a synthetic two-member bundle
  python tools/ci/h5_cross_version.py read DIR [--max-hdf5 MAJOR.MINOR]   # with the library under test

`write` builds the bundle with hcd_analysis.emulator.release_testing.write_synthetic_bundle (small untrained networks,
a made-up eligible lineage block and w_c map: synthetic values only, never a real model) and records, in
NEW_DIR/expected.json, the bundle manifest's sha256, the sha256 of every model leaf and normalizer array, and the
writer's h5py and libhdf5 versions. `read` loads the bundle with safe_h5.load_ensemble anchored on that sha256 (so the
release loader's lineage checks run under the library under test too) and requires every digest to match; with
--max-hdf5 it also requires the reading libhdf5 to be older than that version, so a job meant to test an old library
cannot pass on a new one. Exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _digests(model, norm):
    """{name: sha256 of the array's dtype, shape and bytes} for every model leaf and normalizer array."""
    import jax
    import numpy as np
    from hcd_analysis.emulator import safe_io as S
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(model):
        a = np.asarray(leaf)
        out["model" + jax.tree_util.keystr(path)] = hashlib.sha256(
            f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()
    for name, a in sorted(S.norm_tensors(norm).items()):
        a = np.asarray(a)
        out["norm/" + name] = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()
    return out


def _versions():
    import h5py
    return {"h5py": h5py.version.version, "libhdf5": h5py.version.hdf5_version}


def write(out_dir):
    from hcd_analysis.emulator import release_testing
    os.makedirs(out_dir)
    info = release_testing.write_synthetic_bundle(os.path.join(out_dir, "bundle"))
    expected = {"manifest_sha256": info.manifest_sha256,
                "members": [_digests(model, info.norm) for model in info.models], "writer": _versions()}
    with open(os.path.join(out_dir, "expected.json"), "x") as f:
        json.dump(expected, f, indent=1)
    print(f"wrote {info.bundle_dir} with libhdf5 {expected['writer']['libhdf5']}")
    return 0


def read(in_dir, max_hdf5=None):
    from hcd_analysis.emulator import safe_h5 as H
    with open(os.path.join(in_dir, "expected.json")) as f:
        expected = json.load(f)
    reader = _versions()
    if max_hdf5 is not None:
        have = tuple(int(x) for x in reader["libhdf5"].split(".")[:2])
        limit = tuple(int(x) for x in max_hdf5.split("."))
        if have >= limit:
            print(f"FAIL: reading libhdf5 {reader['libhdf5']} is not older than {max_hdf5}")
            return 1
    ensemble, metas, norm = H.load_ensemble(os.path.join(in_dir, "bundle"),
                                            expected_manifest_sha256=expected["manifest_sha256"])
    problems = []
    if [m["seed"] for m in metas] != [0, 1]:
        problems.append(f"member seeds {[m['seed'] for m in metas]}")
    for i, (model, want) in enumerate(zip(ensemble.members, expected["members"])):
        got = _digests(model, norm)
        problems += [f"member {i} {k}" for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)]
    for p in problems:
        print(f"MISMATCH {p}")
    print(f"read with libhdf5 {reader['libhdf5']} (h5py {reader['h5py']}) a bundle written with libhdf5 "
          f"{expected['writer']['libhdf5']}: {sum(len(m) for m in expected['members'])} arrays, {len(problems)} "
          "mismatches")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("write", "read"))
    ap.add_argument("dir")
    ap.add_argument("--max-hdf5", help="read: require the reading libhdf5 to be older than MAJOR.MINOR")
    args = ap.parse_args(argv)
    return write(args.dir) if args.mode == "write" else read(args.dir, args.max_hdf5)


if __name__ == "__main__":
    sys.exit(main())
