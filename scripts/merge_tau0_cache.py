"""Merge per-shard tau0 caches into one cache (format 3.4).

The production cache is built by sharding the (sim, snap) pairs across many
sbatch tasks (build_emulator_cache_tau0.py --offset/--limit), each writing its
own shard .h5. This script concatenates them:

  - per-row datasets  -> concatenated along axis 0
  - per-snap datasets -> concatenated along axis 0
  - snap_group_idx    -> remapped by the cumulative per-shard snap offset
  - top-level shared datasets (tier_c_labels, tier_c_nhi_edges, param_names,
    log_nhi_*) -> from shard 0 (asserted identical across shards)

Lineage: every shard is read with the release reader (hcd_analysis.tau0_cache.read_cache: format 3.4, its record,
its payload digest, release eligibility); shards must share their settings and may not repeat a group. The merged
record lists each shard and every input the shards' records list. ``--historical-reproduction REASON`` reads 3.3 or
ineligible shards instead and stamps the output not release-eligible. Shards with carried rows are not merged.

Usage:
    python3 scripts/merge_tau0_cache.py --shards 'SHARD_DIR/observables_tau0_*.h5' --output MERGED.h5 \
        [--expect-rows N] [--expect-pairs M] [--historical-reproduction REASON]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_emulator_cache_tau0 as bt0  # noqa: E402
from hcd_analysis import lineage as L  # noqa: E402
from hcd_analysis import tau0_cache as T  # noqa: E402

_TOP_LEVEL = T.STATIC_KEYS
_ROW_STR = ("sim_name",)
_ROW_ARR = (("params",) + T.ROW_P1D_KEYS + T.ROW_TIERC_KEYS + T.ROW_COUNT_KEYS + T.ROW_MEANF_KEYS)
_ROW_FLOAT = T.ROW_FLOAT_KEYS
_ROW_INT = T.ROW_INT_KEYS  # includes snap_group_idx (remapped specially)
_SNAP_STR = ("snap_sim_name",)
_SNAP_INT = ("snap_snap",)
_SNAP_FLOAT = tuple("snap_" + k for k in T.SNAP_FLOAT_KEYS)
_SNAP_2D = tuple("snap_" + k for k in T.SNAP_2D_KEYS)
_REPLACED_ATTRS = ("created_utc", "git_sha", "n_rows", "n_snaps", "merged_from_n_shards", "cache_version")


def read_source(path, historical_reason=None):
    """``(datasets, attrs, record)`` of an input cache: the release reader, or the historical one with a reason."""
    if historical_reason is not None:
        return T.read_cache_historical(path, reason=historical_reason)
    return T.read_cache(path)


def source_entry(role, path, rec):
    """The input entry of a source cache: its file identity, and its record's finder ids and eligibility."""
    ident = L.file_identity(path)
    return L.input_entry(role, T.KIND, {"sha256": ident["sha256"], "size": ident["size"]}, record=rec,
                         info={"path": str(path)})


def groups_of(d, rec):
    """The record's groups, or (historical source without a record) the file's blocks as built groups."""
    if rec is not None and isinstance(rec.get("product", {}).get("groups"), list):
        return [dict(g) for g in rec["product"]["groups"]]
    gi = np.asarray(d["snap_group_idx"])
    return [{"sim": str(s), "snap": int(n), "z_grid": float(np.asarray(d["z_grid"])[np.flatnonzero(gi == b)[0]]),
             "rows": T.ROWS_BUILT} for b, (s, n) in enumerate(zip(d["snap_sim_name"], d["snap_snap"]))]


def merge_shards(shard_paths, output_path, expect_rows=None, expect_pairs=None, *, historical_reason=None):
    shard_paths = sorted(shard_paths)
    if not shard_paths:
        raise ValueError("no shard files matched")
    print(f"Merging {len(shard_paths)} shards -> {output_path}")

    row_data, snap_data, snap_group_idx = {}, {}, []
    n_snaps_so_far = 0
    top, attrs0, settings0 = {}, None, None
    entries, groups = [], []
    for si, sp in enumerate(shard_paths):
        d, attrs, rec = read_source(sp, historical_reason)
        n_rows, n_snaps = len(d["sim_name"]), len(d["snap_snap"])
        print(f"  shard {si}: {Path(sp).name}  n_rows={n_rows} n_snaps={n_snaps}")
        if rec is not None and rec["product"].get("carried"):
            raise ValueError(f"{sp}: a shard with carried rows is not merged")
        if si == 0:
            top = {k: d[k] for k in _TOP_LEVEL}
            attrs0 = attrs
            settings0 = rec["settings"] if rec is not None else None
        else:
            for k in _TOP_LEVEL:
                if not np.array_equal(d[k], top[k]):
                    raise ValueError(f"{k} mismatch in {sp}")
            for a in bt0.RECIPE_ATTRS:
                if not np.array_equal(np.asarray(attrs.get(a)), np.asarray(attrs0.get(a))):
                    raise ValueError(f"attribute {a} mismatch in {sp}")
            if rec is not None and settings0 is not None and rec["settings"] != settings0:
                raise ValueError(f"{sp}: settings differ from the first shard's")
        for k in _ROW_STR + _ROW_ARR + _ROW_FLOAT + _ROW_INT:
            if k == "snap_group_idx":
                snap_group_idx.append(np.asarray(d[k]).astype(np.int64) + n_snaps_so_far)
            else:
                row_data.setdefault(k, []).append(d[k])
        for k in _SNAP_STR + _SNAP_INT + _SNAP_FLOAT + _SNAP_2D:
            snap_data.setdefault(k, []).append(d[k])
        n_snaps_so_far += n_snaps
        groups.extend(groups_of(d, rec))
        entries.append([source_entry(f"shard:{si}", sp, rec if rec is not None and historical_reason is None
                                     else None)] + (list(rec["inputs"]) if rec is not None else []))

    total_rows = sum(len(a) for a in row_data["sim_name"])
    total_snaps = sum(len(a) for a in snap_data["snap_snap"])
    # Completeness guards: a silently failed/OOM'd shard just vanishes from the
    # glob, so the merge would otherwise succeed with missing pairs undetected.
    ids = [(g["sim"], g["snap"]) for g in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("a (sim, snap) group appears in more than one shard")
    n_pairs = len(ids)
    print(f"  merged {total_rows} rows across {n_pairs} unique (sim,snap) pairs "
          f"from {len(shard_paths)} shards")
    if expect_rows is not None and total_rows != expect_rows:
        raise ValueError(f"row-count mismatch: merged {total_rows} != expected {expect_rows} (failed shard?)")
    if expect_pairs is not None and n_pairs != expect_pairs:
        raise ValueError(f"pair-count mismatch: merged {n_pairs} != expected {expect_pairs} (missing shard?)")

    d = dict(top)
    for k, lst in row_data.items():
        d[k] = np.concatenate(lst, axis=0)
    d["snap_group_idx"] = np.concatenate(snap_group_idx).astype(np.int32)
    for k, lst in snap_data.items():
        d[k] = np.concatenate(lst, axis=0)
    gi = d["snap_group_idx"]
    if gi.min() < 0 or gi.max() >= total_snaps:
        raise ValueError(f"snap_group_idx out of range [0,{total_snaps}): [{gi.min()},{gi.max()}]")
    attrs = {k: v for k, v in attrs0.items() if k not in _REPLACED_ATTRS}
    fresh = bt0.cache_attrs(int(attrs0["n_k"]), attrs0["alpha_range"])
    attrs.update(created_utc=fresh["created_utc"], git_sha=fresh["git_sha"], n_rows=total_rows, n_snaps=total_snaps,
                 merged_from_n_shards=len(shard_paths))
    rec = T.write_cache(output_path, d, attrs, inputs=T.combine_inputs(*entries),
                        settings=settings0 if settings0 is not None else {}, groups=groups,
                        historical_reason=historical_reason)
    print(f"Wrote {output_path}  ({Path(output_path).stat().st_size/1e6:.1f} MB, {total_rows} rows, {total_snaps} "
          f"snaps); release_eligible={rec['release_eligible']}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge sharded tau0 caches.")
    ap.add_argument("--shards", required=True,
                    help="glob pattern for shard .h5 files (quote it).")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-rows", type=int, default=None,
                    help="assert merged row count (= n_pairs * n_alpha); catches failed shards.")
    ap.add_argument("--expect-pairs", type=int, default=None,
                    help="assert merged unique (sim,snap) count; catches missing shards.")
    ap.add_argument("--historical-reproduction", metavar="REASON", default=None,
                    help="read shards without lineage checks; the output is stamped not release-eligible")
    args = ap.parse_args(argv)
    merge_shards(glob.glob(args.shards), args.output, expect_rows=args.expect_rows, expect_pairs=args.expect_pairs,
                 historical_reason=args.historical_reproduction)


if __name__ == "__main__":
    main()
