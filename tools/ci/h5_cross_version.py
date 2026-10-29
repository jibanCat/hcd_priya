"""Cross-version check of the HDF5 release format (hcd_priya.h5/1): a bundle written with one libhdf5 is read, bit for
bit, with another (the release writer keeps files readable by libhdf5 1.10).

usage:
  python tools/ci/h5_cross_version.py write NEW_DIR      # with the pinned stack: a synthetic two-member bundle
  python tools/ci/h5_cross_version.py read DIR [--max-hdf5 MAJOR.MINOR]   # with the library under test

`write` records, in NEW_DIR/expected.json, the bundle manifest's sha256, the sha256 of every model leaf and normalizer
array, and the writer's h5py and libhdf5 versions. `read` loads the bundle with safe_h5.load_ensemble (anchored on that
sha256) and requires every digest to match; with --max-hdf5 it also requires the reading libhdf5 to be older than that
version, so a job meant to test an old library cannot pass on a new one. Synthetic values only. Exit status 1 on any
mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ARCH = {"in_dim": 10, "n_k": 8, "n_basis": 4, "baseline_n_layers": 1, "baseline_width": 16}


def _meta(seed):
    from hcd_analysis.emulator.schema import CHECKPOINT_SCHEMA_VERSION
    return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "arch_cfg": dict(ARCH), "seed": seed,
            "k_com_hmpc": [0.05 * (i + 1) for i in range(ARCH["n_k"])], "L_box_hmpc": 120.0,
            "k_convention": "synthetic", "cache_sha256": "0" * 64, "param_limits": [[0.0, 1.0]] * 9,
            "cache_path": "synthetic.h5", "git_sha": "synthetic", "n_k": ARCH["n_k"], "recipe": {"w": 1.0}}


def _norm():
    import numpy as np
    r = np.random.default_rng(0)
    return {"f_nhi": {"mean": r.normal(size=30), "std": r.uniform(1, 2, 30)},
            "dndx": {"mean": r.normal(size=3), "std": r.uniform(1, 2, 3)},
            "P_filt": {"mu_marg": r.normal(size=(4, 8)), "sig_marg": r.uniform(1, 2, (4, 8)),
                       "sig_cosmo": r.uniform(1, 2, (4, 8)),
                       "cell_mean": {i: r.normal(size=(4, 8)) for i in range(5)}},
            "delta": {"mean": r.normal(size=(3, 8)), "std": r.uniform(1, 2, (3, 8))}}


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
    import jax
    from hcd_analysis.emulator import safe_h5 as H
    from hcd_analysis.emulator.model import Emulator
    os.makedirs(out_dir)
    bundle = os.path.join(out_dir, "bundle")
    os.makedirs(bundle)
    prefixes, members = [], []
    for i in range(2):
        model = Emulator(**ARCH, key=jax.random.PRNGKey(i))
        prefix = os.path.join(bundle, f"m{i}")
        H.export_member(prefix, model, _meta(i), _norm())
        prefixes.append(prefix)
        members.append(_digests(model, _norm()))
    H.write_manifest(bundle, prefixes, {"source": "synthetic cross-version check"})
    expected = {"manifest_sha256": H.sha256_file(os.path.join(bundle, "manifest.json")), "members": members,
                "writer": _versions()}
    with open(os.path.join(out_dir, "expected.json"), "x") as f:
        json.dump(expected, f, indent=1)
    print(f"wrote {bundle} with libhdf5 {expected['writer']['libhdf5']}")
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
