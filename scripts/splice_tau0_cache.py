"""Splice rebuilt (sim, snap) groups into a tau0 cache, and audit two caches group by group.

Used for the S4/F8 repair (emulator-debug campaign 2026-10; spec hcd_priya_notes
docs/superpowers/emulator-debug-2026-10/gateB/S4_REPAIR_SPEC.md): the historical cache is never modified; a NEW file is
written in which named groups are removed and the groups of freshly built shards are inserted, in the canonical
(sim, snap) order the builder produces, with snap_group_idx recomputed. `cache_delta` compares two caches by identity
(row = (sim, z_grid, alpha_idx); snapshot block = (sim, snap)) so position shifts are not mistaken for changes.

Usage (<repo> = this checkout):
    python <repo>/scripts/splice_tau0_cache.py --base OLD.h5 --shard NEW_GROUPS.h5 [--shard ...] \
        --remove SIM:SNAP:ZGRID [--remove ...] --record-json RECORD.json --output REPAIRED.h5
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import merge_tau0_cache as mg  # noqa: E402

ROW_KEYS = mg._ROW_STR + mg._ROW_ARR + mg._ROW_FLOAT + mg._ROW_INT
SNAP_KEYS = mg._SNAP_STR + mg._SNAP_INT + mg._SNAP_FLOAT + mg._SNAP_2D
STATIC_KEYS = mg._TOP_LEVEL
ALL_KEYS = set(ROW_KEYS) | set(SNAP_KEYS) | set(STATIC_KEYS)
STR_KEYS = {"sim_name", "snap_sim_name", "tier_c_labels", "param_names"}
_REPLACED_ATTRS = ("created_utc", "git_sha", "n_rows", "n_snaps", "merged_from_n_shards", "repair_record")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if keys != ALL_KEYS:
            raise ValueError(f"{path}: key set differs from the tau0 cache layout: "
                             f"missing {sorted(ALL_KEYS - keys)}, unknown {sorted(keys - ALL_KEYS)}")
        d = {k: (f[k].asstr()[...] if k in STR_KEYS else f[k][...]) for k in keys}
        dt = {k: f[k].dtype for k in keys}
        attrs = dict(f.attrs)
    return d, dt, attrs


def _groups(d, path):
    """[(identity (sim, snap), block index, row indices)] in file order; checks rows point at their own block."""
    out = []
    gi = d["snap_group_idx"]
    for b, (sim, snap) in enumerate(zip(d["snap_sim_name"], d["snap_snap"])):
        rows = np.where(gi == b)[0]
        if rows.size == 0 or np.any(d["sim_name"][rows] != sim) or np.any(d["snap"][rows] != snap):
            raise ValueError(f"{path}: snapshot block {b} ({sim}, {snap}) and its rows disagree")
        out.append(((str(sim), int(snap)), b, rows))
    if sum(r.size for _, _, r in out) != d["sim_name"].size:
        raise ValueError(f"{path}: rows without a snapshot block")
    return out


def _check_canonical(groups, d, path):
    ids = [g[0] for g in groups]
    if ids != sorted(ids):
        raise ValueError(f"{path}: snapshot blocks are not in canonical (sim, snap) order")
    order = np.concatenate([r for _, _, r in groups])
    if not np.array_equal(order, np.arange(order.size)):
        raise ValueError(f"{path}: rows are not in canonical (sim, snap) order")
    for _, _, r in groups:
        if not np.array_equal(d["alpha_idx"][r], np.sort(d["alpha_idx"][r])):
            raise ValueError(f"{path}: rows of a group are not in canonical alpha order")


def splice_cache(base_path, shard_paths, remove, out_path, *, record, code_commit=None):
    """Write a NEW cache = base minus the groups `remove` [(sim, snap, z_grid)] plus every group of `shard_paths`."""
    out_path = Path(out_path).resolve()
    if out_path in {Path(p).resolve() for p in [base_path, *shard_paths]}:
        raise ValueError("refusing to overwrite an input cache")
    base, base_dt, base_attrs = _load(base_path)
    bgroups = _groups(base, base_path)
    _check_canonical(bgroups, base, base_path)
    keep = {g[0]: ("base", g[1], g[2]) for g in bgroups}
    removed = []
    for sim, snap, zg in remove:
        g = keep.get((sim, int(snap)))
        if g is None or not np.all(np.abs(base["z_grid"][g[2]] - float(zg)) < 1e-9):
            raise ValueError(f"group to remove not found: {(sim, snap, zg)}")
        del keep[(sim, int(snap))]
        removed.append([sim, int(snap), float(zg)])
    sources = {"base": base}
    added, shard_meta = [], []
    for i, sp in enumerate(shard_paths):
        d, dt, attrs = _load(sp)
        for k in STATIC_KEYS:
            if not np.array_equal(d[k], base[k]):
                raise ValueError(f"shard {sp}: static table {k} differs from the base")
        for a in ("n_k", "tier_c_recipe", "cache_version", "tau_thresh"):
            if str(attrs.get(a)) != str(base_attrs.get(a)):
                raise ValueError(f"shard {sp}: attribute {a} {attrs.get(a)!r} != base {base_attrs.get(a)!r}")
        if not np.allclose(attrs["alpha_range"], base_attrs["alpha_range"], rtol=0, atol=0):
            raise ValueError(f"shard {sp}: alpha_range differs from the base")
        for k in base_dt:
            if dt[k].kind != base_dt[k].kind:
                raise ValueError(f"shard {sp}: dtype kind of {k} differs from the base")
        src = f"shard{i}"
        sources[src] = d
        for ident, b, rows in _groups(d, sp):
            if ident in keep:
                raise ValueError(f"group {ident} from {sp} already present in the cache")
            keep[ident] = (src, b, rows)
            added.append([ident[0], ident[1], float(d["z_grid"][rows[0]])])
        shard_meta.append({"path": str(sp), "sha256": sha256(sp)})

    order = sorted(keep)
    rows_out = {k: [] for k in ROW_KEYS}
    snap_out = {k: [] for k in SNAP_KEYS}
    for gnew, ident in enumerate(order):
        src, b, rows = keep[ident]
        d = sources[src]
        for k in ROW_KEYS:
            rows_out[k].append(np.full(rows.size, gnew) if k == "snap_group_idx" else d[k][rows])
        for k in SNAP_KEYS:
            snap_out[k].append(d[k][b:b + 1])
    n_rows = sum(len(x) for x in rows_out["sim_name"])
    rec = dict(record)
    rec.update(base=str(base_path), base_sha256=sha256(base_path), shards=shard_meta, removed=removed, added=added,
               n_rows=[int(base_attrs["n_rows"]), n_rows], n_snaps=[int(base_attrs["n_snaps"]), len(order)],
               code_commit=code_commit)
    with h5py.File(out_path, "w") as f:
        for k, v in base_attrs.items():
            if k not in _REPLACED_ATTRS:
                f.attrs[k] = v
        f.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        f.attrs["git_sha"] = code_commit or "unknown"
        f.attrs["n_rows"] = n_rows
        f.attrs["n_snaps"] = len(order)
        f.attrs["repair_record"] = json.dumps(rec, sort_keys=True)
        for k in STATIC_KEYS:
            _write(f, k, base[k], base_dt[k])
        for k in ROW_KEYS:
            _write(f, k, np.concatenate(rows_out[k], axis=0), base_dt[k])
        for k in SNAP_KEYS:
            _write(f, k, np.concatenate(snap_out[k], axis=0), base_dt[k])
    return rec


def _write(f, k, arr, dtype):
    if k in STR_KEYS:
        f.create_dataset(k, data=np.asarray(arr, dtype=object), dtype=h5py.string_dtype())
    else:
        f.create_dataset(k, data=np.asarray(arr).astype(dtype))


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0.0
    den = np.maximum(np.abs(a[m]), np.abs(b[m]))
    return float(np.max(np.where(den > 0, np.abs(a[m] - b[m]) / np.where(den > 0, den, 1), 0.0)))


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def cache_delta(hist_path, new_path):
    """Identity-matched comparison of two tau0 caches. Every per-row key except snap_group_idx is compared exactly
    (NaN == NaN); snap_group_idx is compared through the identity of the block it points at."""
    H, Hdt, Ha = _load(hist_path)
    N, Ndt, Na = _load(new_path)
    rep = {"key_set_equal": True, "dtypes_equal": all(Hdt[k] == Ndt[k] for k in Hdt)}
    rep["static_diffs"] = [k for k in STATIC_KEYS if not _same(H[k], N[k])]
    rep["static_equal"] = not rep["static_diffs"]

    def rows_by_id(d):
        ids = list(zip(d["sim_name"].tolist(), np.round(d["z_grid"], 6).tolist(), d["alpha_idx"].tolist()))
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate row identities")
        return {i: r for r, i in enumerate(ids)}

    def blocks_by_id(d):
        ids = list(zip(d["snap_sim_name"].tolist(), d["snap_snap"].tolist()))
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate block identities")
        return {i: b for b, i in enumerate(ids)}

    hr, nr, hb, nb = rows_by_id(H), rows_by_id(N), blocks_by_id(H), blocks_by_id(N)
    common = sorted(set(hr) & set(nr))
    rep["row_identities_common"] = common
    rep["rows_only_hist"] = sorted(set(hr) - set(nr))
    rep["rows_only_new"] = sorted(set(nr) - set(hr))
    changed = {}
    for i in common:
        a, b = hr[i], nr[i]
        diff = {}
        for k in ROW_KEYS:
            if k == "snap_group_idx":
                ha = (H["snap_sim_name"][H[k][a]], int(H["snap_snap"][H[k][a]]))
                na = (N["snap_sim_name"][N[k][b]], int(N["snap_snap"][N[k][b]]))
                if ha != na:
                    diff[k] = f"block {ha} -> {na}"
            elif not _same(H[k][a], N[k][b]):
                diff[k] = _rel(H[k][a], N[k][b]) if H[k].dtype.kind in "fiu" else "differs"
        if diff:
            changed[i] = diff
    rep["rows_changed"] = changed
    rep["blocks_only_hist"] = sorted(set(hb) - set(nb))
    rep["blocks_only_new"] = sorted(set(nb) - set(hb))
    bchanged = {}
    for i in sorted(set(hb) & set(nb)):
        diff = [k for k in SNAP_KEYS if not _same(H[k][hb[i]], N[k][nb[i]])]
        if diff:
            bchanged[i] = diff
    rep["blocks_changed"] = bchanged
    rep["attrs_changed"] = sorted(k for k in set(Ha) | set(Na) if str(Ha.get(k)) != str(Na.get(k)))
    return rep


def main():
    ap = argparse.ArgumentParser(description="Splice rebuilt groups into a NEW tau0 cache.")
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--shard", action="append", required=True, type=Path)
    ap.add_argument("--remove", action="append", default=[], metavar="SIM:SNAP:ZGRID")
    ap.add_argument("--record-json", type=Path, required=True)
    ap.add_argument("--code-commit", default=None)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    remove = []
    for r in a.remove:
        sim, snap, zg = r.rsplit(":", 2)
        remove.append((sim, int(snap), float(zg)))
    rec = splice_cache(a.base, a.shard, remove, a.output, record=json.load(open(a.record_json)),
                       code_commit=a.code_commit)
    print(json.dumps({k: rec[k] for k in ("removed", "added", "n_rows", "n_snaps", "base_sha256")}, indent=1))
    print(f"wrote {a.output} sha256 {sha256(a.output)}")


if __name__ == "__main__":
    main()
