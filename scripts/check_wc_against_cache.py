#!/usr/bin/env python3
"""The w_c acceptance check of a release bundle: the class fractions the public predictor returns (the members' mean
Head-A dN/dX through the bundled w_c map) against the training cache's own sightline fractions, at every simulation of
the cache and the given redshifts, with a relative tolerance per class fixed on the command line before the comparison.

The cache must be the bundle's: its sha256 is checked against ``--cache-sha256`` and against the bundle's lineage block
(the export read it through the strict release reader), so only the few datasets needed are read here (the per-row
Tier-C counts, the parameters, the redshifts, the group index and the simulation names), and the file must not change
while they are read. Per (simulation, z): the cache fraction of class c is its coarse sightline count over the total,
from the group's rows (which must agree); the prediction is ``predict(params, z, tau_eff=tau_Kim07(z)).w_c`` (Head A
does not depend on the mean flux). The relative error |w_pred / w_cache - 1| must be within the class tolerance for every
pair, and every simulation must have every requested redshift. Writes a JSON report; exit 0 PASS, 1 FAIL, 2 refused.

    python scripts/check_wc_against_cache.py --bundle DIR --manifest-sha256 HEX --cache CACHE.h5 --cache-sha256 HEX \\
        --z 2.6 2.8 ... --tol LLS=0.02 --tol subDLA=0.02 --tol DLA=0.05 --out REPORT.json
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import hcd_analysis.predictor as hp  # noqa: E402
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis.emulator.meanflux_prior import kim_tau0  # noqa: E402
from hcd_analysis.tau0_cache import _collapse_counts  # noqa: E402

KEYS = ("tier_c_counts", "params", "z_grid", "snap_group_idx", "sim_name")


def read_cache_columns(path, sha256):
    """The datasets the check needs, from the file whose sha256 is ``sha256`` (unchanged while read)."""
    import h5py
    ident = L.file_identity(path)
    if ident["sha256"] != sha256:
        raise SystemExit(f"refusing: {path} has sha256 {ident['sha256']}, not {sha256}")
    with h5py.File(path, "r") as f:
        d = {k: (f[k].asstr()[...] if k == "sim_name" else f[k][...]) for k in KEYS}
    st = os.stat(path)
    if (int(st.st_size), int(st.st_mtime_ns)) != (ident["size"], ident["mtime_ns"]):
        raise SystemExit(f"refusing: {path} changed while it was read")
    return d


def cache_pairs(d, zs, tol_z=1e-6):
    """{(sim, z): (params (9,), cache w_c (3,) LLS, subDLA, DLA)} for every group at a requested z."""
    cc = _collapse_counts(np.asarray(d["tier_c_counts"]))
    out = {}
    for g in np.unique(d["snap_group_idx"]):
        rows = np.flatnonzero(d["snap_group_idx"] == g)
        z = float(d["z_grid"][rows[0]])
        hit = [zz for zz in zs if abs(z - zz) < tol_z]
        if not hit:
            continue
        if not (np.all(cc[rows] == cc[rows[0]]) and np.all(d["params"][rows] == d["params"][rows[0]])):
            raise SystemExit(f"refusing: group {g} has rows with different counts or parameters")
        n = cc[rows[0]].sum()
        key = (str(d["sim_name"][rows[0]]), hit[0])
        if key in out:
            raise SystemExit(f"refusing: two groups for {key}")
        out[key] = (np.asarray(d["params"][rows[0]], dtype=np.float64), cc[rows[0], 1:] / n)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cache-sha256", required=True)
    ap.add_argument("--z", type=float, nargs="+", required=True)
    ap.add_argument("--tol", action="append", required=True, metavar="CLASS=REL",
                    help="relative tolerance per class (LLS, subDLA, DLA), each exactly once")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tol = {}
    for t in a.tol:
        c, _, v = t.partition("=")
        if c not in hp.HCD_CLASSES or c in tol:
            raise SystemExit(f"refusing: --tol {t!r}")
        tol[c] = float(v)
    if set(tol) != set(hp.HCD_CLASSES):
        raise SystemExit(f"refusing: a tolerance for each of {hp.HCD_CLASSES} is needed")
    if os.path.exists(a.out):
        raise SystemExit(f"refusing: {a.out} exists")
    emu = hp.load_bundle(a.bundle, a.manifest_sha256)
    if emu.lineage["cache"]["sha256"] != a.cache_sha256:
        raise SystemExit(f"refusing: the bundle was trained on the cache {emu.lineage['cache']['sha256']}, not "
                         f"{a.cache_sha256}")
    zs = [hp.snapshot_z(z) for z in a.z]
    pairs = cache_pairs(read_cache_columns(a.cache, a.cache_sha256), zs)
    sims = sorted({s for s, _ in pairs})
    missing = [f"{s} {z}" for s in sims for z in zs if (s, z) not in pairs]
    rows = []
    for (sim, z), (pv, w_cache) in sorted(pairs.items()):
        params = dict(zip(hp.PARAM_NAMES, pv.tolist()))
        w = emu.predict(params, z, tau_eff=float(kim_tau0(z))).w_c
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(w_cache > 0, np.abs(w / w_cache - 1.0), np.inf)
        rows.append({"sim": sim, "z": z, "w_pred": w.tolist(), "w_cache": w_cache.tolist(), "rel": rel.tolist(),
                     "ok": bool(all(r <= tol[c] for r, c in zip(rel, hp.HCD_CLASSES)))})
    per = {}
    for i, c in enumerate(hp.HCD_CLASSES):
        r = np.array([x["rel"][i] for x in rows])
        worst = rows[int(np.argmax(r))] if r.size else None
        per[c] = {"tol": tol[c], "max_rel": float(r.max()) if r.size else None,
                  "median_rel": float(np.median(r)) if r.size else None, "n_over": int(np.sum(r > tol[c])),
                  "worst": None if worst is None else {"sim": worst["sim"], "z": worst["z"]}}
    passed = bool(rows) and not missing and all(x["ok"] for x in rows)
    report = {"format": "hcd_priya.wc_acceptance/1", "passed": passed, "bundle_manifest_sha256": a.manifest_sha256,
              "cache_sha256": a.cache_sha256, "wc_map_sha256": emu.wc_map.sha256, "z": zs, "n_sims": len(sims),
              "n_pairs": len(rows), "missing": missing, "per_class": per, "pairs": rows, "code": L.code_identity(),
              "rule": "|w_pred / w_cache - 1| <= tol per class at every (simulation, z); w_pred from the public "
                      "predictor at tau_eff = tau_Kim07(z) (Head A does not depend on the mean flux)"}
    with open(a.out, "x") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"{'PASSED' if passed else 'FAILED'}: {len(rows)} (simulation, z) pairs, {len(sims)} simulations, "
          f"{len(missing)} missing")
    for c, v in per.items():
        print(f"  {c:7s} max rel {v['max_rel']}  median {v['median_rel']}  over tol {v['n_over']} (tol {v['tol']})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
